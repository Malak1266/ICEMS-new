"""
ICEMS V3 — Temporal Hierarchical Scorer
Architecture : Conv1D Multi-échelle + BiLSTM + Transformer + GRU Causal

Deux têtes :
- Score continu [-1, +1] (principale)
- Logits 9 sous-niveaux (auxiliaire, supervision fine)
"""

import math

import torch
import torch.nn as nn


class MultiScaleConvEncoder(nn.Module):
    """
    Encodeur convolutif multi-échelle.
    Capture les patterns à 3 échelles temporelles :
    - Micro  (kernel=3)  : tremblements, micro-hésitations
    - Local  (kernel=7)  : transitions entre phases
    - Global (kernel=15) : cohérence de trajectoire
    """

    def __init__(self, n_features=6, d_out=128):
        super().__init__()
        d_each = d_out // 3

        self.conv_micro = nn.Sequential(
            nn.Conv1d(n_features, d_each, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_each), nn.ReLU(),
        )
        self.conv_local = nn.Sequential(
            nn.Conv1d(n_features, d_each, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_each), nn.ReLU(),
        )
        self.conv_global = nn.Sequential(
            nn.Conv1d(n_features, d_each, kernel_size=15, padding=7),
            nn.BatchNorm1d(d_each), nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_each * 3, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU(),
        )

    def forward(self, x):
        xt = x.transpose(1, 2)
        m = self.conv_micro(xt).transpose(1, 2)
        l = self.conv_local(xt).transpose(1, 2)
        g = self.conv_global(xt).transpose(1, 2)
        return self.fusion(torch.cat([m, l, g], dim=-1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ICEMS_V3(nn.Module):
    """
    Architecture complète ICEMS V3.

    Flux :
    [B, T, 6]
    → MultiScaleConv [B, T, 128]
    → BiLSTM [B, T, 128]
    → Transformer [B, T, 128]
    → Fusion + LayerNorm [B, T, 128]
    → GRU Causal [B, T, 64]
    → Score [-1,+1] par frame [B, T]
    → Score agrégé [B]
    → Logits 4 classes [B, 4]
    → Logits 9 sous-niveaux [B, 9]
    """

    def __init__(
        self,
        n_features=6,
        d_model=128,
        n_heads=4,
        n_transformer_layers=3,
        n_lstm_layers=2,
        n_gru_layers=2,
        dropout=0.3,
        n_classes_coarse=4,
        n_classes_fine=9,
    ):
        super().__init__()

        self.conv_encoder = MultiScaleConvEncoder(n_features, d_model)

        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_transformer_layers)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.causal_gru = nn.GRU(
            input_size=d_model,
            hidden_size=64,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        self.score_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Tanh(),
        )

        self.coarse_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes_coarse),
        )

        self.fine_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes_fine),
        )

    def forward(self, x, mask=None):
        """
        x    : [B, T, 6]
        mask : [B, T] bool, True = frame valide

        Retourne :
        scores_per_frame : [B, T]
        score_agg        : [B]
        logits_4         : [B, 4]
        logits_9         : [B, 9]
        gru_hidden       : [B, 64]  pour t-SNE
        """
        h = self.conv_encoder(x)

        lstm_out, _ = self.bilstm(h)

        h_pos = self.pos_enc(h)
        padding_mask = ~mask if mask is not None else None
        tf_out = self.transformer(h_pos, src_key_padding_mask=padding_mask)

        fused = self.fusion(torch.cat([lstm_out, tf_out], dim=-1))

        gru_out, _ = self.causal_gru(fused)

        scores = self.score_head(gru_out).squeeze(-1)

        if mask is not None:
            m = mask.float()
            score_agg = (scores * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            score_agg = scores.mean(1)

        last_hidden = gru_out[:, -1, :]
        logits_4 = self.coarse_head(last_hidden)
        logits_9 = self.fine_head(last_hidden)

        return scores, score_agg, logits_4, logits_9, last_hidden
