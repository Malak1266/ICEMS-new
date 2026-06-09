"""
step_A_data_generation.py
=========================
GRU conditionnel pour générer des séquences synthétiques à partir des trials labellisés.

Structure attendue de continuous_per_trial.pkl (vérifiée) :
    {(participant_id, trial_id): {
        "X": ndarray (T, 10), "y9": int, "y_reg": float,
        "level": str, "T": int, "fs": float}, ...}

Mapping 4 classes (à partir de y9) :
    0 Student (y9=0), 1 Junior (y9=1..5), 2 Senior (y9=6..7), 3 Expert (y9=8)

Usage (depuis ICEMS-main) :
    python src/step_A_data_generation.py
    python src/step_A_data_generation.py --input data/continuous_per_trial.pkl --out data/augmented_trials.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# ── Mapping 4 niveaux d'expertise ───────────────────────────────────────────
Y9_TO_Y4 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
Y4_TO_REG = {0: -1.0, 1: -0.33, 2: 0.33, 3: 1.0}
Y4_TO_Y9 = {0: 0, 1: 3, 2: 7, 3: 8}
Y4_TO_LEVEL = {
    0: "Medical student",
    1: "Resident PGY3",
    2: "Fellow",
    3: "Staff",
}
CLASS4_NAMES = ["Student", "Junior", "Senior", "Expert"]

N_FEATURES = 10
EMBED_DIM = 8
HIDDEN_SIZE = 64
SEED_FRAMES = 10
GEN_PER_CLASS = 80
NOISE_STD = 0.05
VELOCITY_CH = 0
TRAIN_CROP_LEN = 512   # fenêtre aléatoire (évite GRU sur 10k+ frames)
GEN_MAX_LEN = 2000       # plafond génération (≈ longueur utile à 10 Hz)


def add_y4_fields(rec: dict) -> dict:
    """Ajoute y4 / y4_reg cohérents avec le barème 4 classes."""
    y4 = Y9_TO_Y4[int(rec["y9"])]
    out = dict(rec)
    out["y4"] = y4
    out["y4_reg"] = Y4_TO_REG[y4]
    return out


class ConditionalGRUGenerator(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, n_classes: int = 4):
        super().__init__()
        self.embed = nn.Embedding(n_classes, EMBED_DIM)
        self.gru = nn.GRU(
            n_features + EMBED_DIM,
            HIDDEN_SIZE,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.GELU(),
            nn.Linear(32, n_features),
        )

    def forward(self, x: torch.Tensor, y4: torch.Tensor) -> torch.Tensor:
        """Teacher forcing : prédit frame t+1 pour chaque position t."""
        emb = self.embed(y4).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.gru(torch.cat([x, emb], dim=-1))
        return self.head(out)

    @torch.no_grad()
    def generate_sequence(
        self,
        seed: np.ndarray,
        y4: int,
        length: int,
        noise_std: float = NOISE_STD,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Génère autoregressivement à partir des SEED_FRAMES premières frames."""
        device = device or next(self.parameters()).device
        self.eval()
        frames = seed[:SEED_FRAMES].astype(np.float32).copy()
        if frames.shape[0] < SEED_FRAMES:
            pad = np.tile(frames[-1:], (SEED_FRAMES - frames.shape[0], 1))
            frames = np.vstack([frames, pad])

        y_t = torch.tensor([y4], dtype=torch.long, device=device)
        emb = self.embed(y_t)  # (1, 8)
        h = None

        for _ in range(SEED_FRAMES):
            x_t = torch.tensor(frames[-1:], dtype=torch.float32, device=device).unsqueeze(0)
            inp = torch.cat([x_t, emb.unsqueeze(1)], dim=-1)
            _, h = self.gru(inp, h)

        while len(frames) < length:
            x_t = torch.tensor(frames[-1:], dtype=torch.float32, device=device).unsqueeze(0)
            inp = torch.cat([x_t, emb.unsqueeze(1)], dim=-1)
            out, h = self.gru(inp, h)
            nxt = self.head(out).squeeze(0).cpu().numpy()
            if noise_std > 0:
                nxt = nxt + np.random.randn(*nxt.shape).astype(np.float32) * noise_std
            frames = np.vstack([frames, nxt.reshape(1, -1)])

        return frames[:length].astype(np.float32)


class TrialSeqDataset(Dataset):
    """Trials entiers ; extrait une fenêtre aléatoire à chaque accès."""

    def __init__(self, trials: List[Tuple[np.ndarray, int]], crop_len: int = TRAIN_CROP_LEN):
        self.items = []
        self.crop_len = crop_len
        for x, y4 in trials:
            if x.shape[0] >= 2:
                self.items.append((x.astype(np.float32), y4))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, y4 = self.items[idx]
        T = x.shape[0]
        if T > self.crop_len:
            start = np.random.randint(0, T - self.crop_len)
            x = x[start : start + self.crop_len]
        return torch.from_numpy(x.copy()), y4, x.shape[0]


def collate_pad(batch):
    lengths = [b[2] for b in batch]
    T_max = max(lengths)
    B = len(batch)
    xs = torch.zeros(B, T_max, N_FEATURES)
    y4s = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, (x, y4, L) in enumerate(batch):
        xs[i, :L] = x
        y4s[i] = y4
        mask[i, :L] = True
    return xs, y4s, mask


def train_model(
    model: ConditionalGRUGenerator,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
) -> List[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for ep in range(1, epochs + 1):
        ep_loss, n = 0.0, 0
        for xs, y4s, mask in loader:
            xs, y4s, mask = xs.to(device), y4s.to(device), mask.to(device)
            pred = model(xs, y4s)
            target = xs[:, 1:, :]
            pred = pred[:, :-1, :]
            m = mask[:, 1:]
            if not m.any():
                continue
            loss = ((pred - target) ** 2)[m.unsqueeze(-1).expand_as(pred)].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n += 1
        avg = ep_loss / max(n, 1)
        losses.append(avg)
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:>2}/{epochs}  loss={avg:.6f}")
    return losses


def mean_length_by_class(dataset: Dict[Tuple[str, str], dict]) -> Dict[int, int]:
    lengths = defaultdict(list)
    for rec in dataset.values():
        y4 = Y9_TO_Y4[int(rec["y9"])]
        lengths[y4].append(int(rec["T"]))
    return {c: int(np.mean(lengths[c])) for c in range(4)}


def validate_velocity(real_by_c: dict, synth_by_c: dict) -> None:
    print("\n[Validation vitesse moyenne — canal 0 bipolar.velocity]")
    for c, name in enumerate(CLASS4_NAMES):
        r = real_by_c.get(c, [])
        s = synth_by_c.get(c, [])
        if not r or not s:
            continue
        mr, ms = float(np.mean(r)), float(np.mean(s))
        if abs(mr) < 1e-9:
            err_pct = 0.0 if abs(ms) < 1e-9 else 100.0
        else:
            err_pct = abs(ms - mr) / abs(mr) * 100.0
        flag = " ⚠️" if err_pct > 20.0 else ""
        print(f"  {name:>7}: réel={mr:.4f}  synth={ms:.4f}  écart={err_pct:.1f}%{flag}")


def count_by_y4(dataset: Dict) -> Counter:
    return Counter(Y9_TO_Y4[int(v["y9"])] for v in dataset.values())


def main():
    ap = argparse.ArgumentParser(description="Step A — génération GRU conditionnelle.")
    ap.add_argument("--input", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out", type=Path, default=Path("data/augmented_trials.pkl"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=GEN_PER_CLASS)
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} introuvable.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.input, "rb") as f:
        raw = pickle.load(f)

    print("=" * 60)
    print(" Step A — Conditional GRU data generation")
    print("=" * 60)
    print(f"\n[Structure pkl] dict, {len(raw)} trials")
    k0 = next(iter(raw.keys()))
    print(f"  clé exemple: {k0} → champs: {list(raw[k0].keys())}")
    print(f"  X shape exemple: {raw[k0]['X'].shape}")

    dataset = {k: add_y4_fields(v) for k, v in raw.items()}
    before = count_by_y4(dataset)
    print("\n[Bilan AVANT augmentation — trials par classe y4]")
    for c in range(4):
        print(f"  {CLASS4_NAMES[c]:>7} (y4={c}): {before[c]:>3}")

    trials_train = [
        (rec["X"], Y9_TO_Y4[int(rec["y9"])])
        for rec in dataset.values()
        if rec["X"].shape[0] >= 2
    ]
    mean_len = mean_length_by_class(dataset)
    print(f"\n[Longueurs moyennes par classe] {mean_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Entraînement] device={device}, epochs={args.epochs}")
    ds = TrialSeqDataset(trials_train, crop_len=TRAIN_CROP_LEN)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)

    model = ConditionalGRUGenerator().to(device)
    train_model(model, loader, device, epochs=args.epochs)

    # Références vitesse réelles par classe
    real_vel = defaultdict(list)
    seeds_by_class = defaultdict(list)
    for rec in dataset.values():
        y4 = Y9_TO_Y4[int(rec["y9"])]
        real_vel[y4].append(float(rec["X"][:, VELOCITY_CH].mean()))
        if rec["X"].shape[0] >= SEED_FRAMES:
            seeds_by_class[y4].append(rec["X"])

    augmented = dict(dataset)
    synth_vel = defaultdict(list)
    synth_count = Counter()

    print(f"\n[Génération] {args.per_class} séquences / classe")
    for y4 in range(4):
        if not seeds_by_class[y4]:
            print(f"  ⚠️ Pas de graine pour {CLASS4_NAMES[y4]}, skip.")
            continue
        target_len = min(mean_len[y4], GEN_MAX_LEN)
        for i in range(args.per_class):
            seed = seeds_by_class[y4][np.random.randint(len(seeds_by_class[y4]))]
            seq = model.generate_sequence(seed, y4, target_len, device=device)
            synth_vel[y4].append(float(seq[:, VELOCITY_CH].mean()))
            key = (f"synth_y4{y4}_{i:03d}", f"Trial{i + 1}")
            fs_med = float(np.median([v["fs"] for v in dataset.values() if Y9_TO_Y4[int(v["y9"])] == y4]))
            augmented[key] = {
                "X": seq,
                "y9": Y4_TO_Y9[y4],
                "y_reg": Y4_TO_REG[y4],
                "y4": y4,
                "y4_reg": Y4_TO_REG[y4],
                "level": Y4_TO_LEVEL[y4],
                "T": int(seq.shape[0]),
                "fs": fs_med,
                "synthetic": True,
            }
            synth_count[y4] += 1

    validate_velocity(real_vel, synth_vel)
    after = count_by_y4(augmented)

    print("\n[Bilan APRÈS augmentation — trials par classe y4]")
    for c in range(4):
        print(f"  {CLASS4_NAMES[c]:>7}: {before[c]:>3} → {after[c]:>3}  (+{after[c] - before[c]})")
    print(f"\n  Total trials: {len(dataset)} → {len(augmented)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(augmented, f)
    print(f"\n✅ Dataset sauvegardé : {args.out.resolve()}")


if __name__ == "__main__":
    main()
