# -*- coding: utf-8 -*-
"""
CV 5-fold par participants — LSTM+Transformer — Features (pos + dist) — Nettoyage outliers
- Fenêtrage: SEQ_LEN=100, HOP=SEQ_LEN (pas de chevauchement)
- Rejet des fenêtres contenant >=50% de zéros (sur données brutes sélectionnées, avant nettoyage)
- Nettoyage outliers par trial: MAD-zscore -> interpolation petits runs -> filtre médian -> clipping -> z-score
- Features gardées (via meta): X,Y,Z, Position Magnitude, Dist Bip-Cavitron, Dist Bip-Scissors
- Backbone: Dense(embed) -> BiLSTM -> (N_BLOCKS x TransformerBlock) -> GAP -> Dropout
- Multi-tâches: coarse (4) + fine/PGY (6), masque fine pour non-résidents (train + val)
- Équilibrage train: sur-échantillonnage par classe 9-niveaux (après split fold)
- Monitoring: balanced accuracy 9-classes (recomposée)
- Confusions:
    * confusion_windows_oof.png (par niveau, fenêtres)
    * confusion_participants_oof.png (par individu × classes 9-niveaux, fenêtres)
    * confusion_participants_majority_oof.png (par niveau, individus agrégés par majorité)
"""

import os, pickle, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import median_filter
from sklearn.model_selection import KFold

# =========================
# Chemins & réglages
# =========================
DATA_PATH = r"./01_10_2025/data/final_from_full_A_plus_dist_levels.pkl"
META_PATH = os.path.splitext(DATA_PATH)[0] + "_meta.json"
OUT_DIR   = "./01_10_2025/runs_kfold5_posdist_outlier_clean"

SEQ_LEN    = 100
HOP        = SEQ_LEN   # pas de chevauchement
BATCH_SIZE = 64
EPOCHS     = 60
SEED       = 42

# Modèle
EMBED_DIM   = 128
LSTM_UNITS  = EMBED_DIM // 2
NUM_HEADS   = 4
FF_DIM      = 128
DROPOUT     = 0.20
N_BLOCKS    = 2

# Optim
BASE_LR      = 1e-3
WEIGHT_DECAY = 3e-4
CLIP_NORM    = 1.0

# Nettoyage outliers
MAD_THRESH         = 6.0   # seuil z-score robuste
MAX_INTERP_RUN     = 20    # longueur max d'un run à interpoler
MEDIAN_WIN         = 3     # filtre médian léger
CLIP_SIGMA         = 5.0   # clipping doux ±5σ

# Filtre des fenêtres à 0
ZERO_RATIO_THRESHOLD = 0.50  # rejeter si >= 50% de zéros

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Labels
# =========================
LEVEL9_ORDER = [
    "Medical student","Resident PGY1","Resident PGY2","Resident PGY3",
    "Resident PGY4","Resident PGY5","Resident PGY6","Fellow","Staff",
]
LEVEL_TO_IDX = {n:i for i,n in enumerate(LEVEL9_ORDER)}
COARSE_ORDER = ["Student","Resident","Fellow","Staff"]
COARSE_TO_IDX = {k:i for i,k in enumerate(COARSE_ORDER)}
PGY_ORDER = ["PGY1","PGY2","PGY3","PGY4","PGY5","PGY6"]
PGY_TO_IDX = {k:i for i,k in enumerate(PGY_ORDER)}

def normalize_spaces(s: str) -> str:
    return " ".join(str(s).replace("_"," ").replace("/", " ").split())

def map_level_to_9(raw: str) -> str:
    if not isinstance(raw, str): return None
    s = normalize_spaces(raw).strip().lower()
    if s.startswith("medical student"): return "Medical student"
    if s.startswith("resident"):
        s2 = s.replace(" ", "")
        for k in range(1,7):
            if f"pgy{k}" in s2: return f"Resident PGY{k}"
        return None
    if s.startswith("fellow"): return "Fellow"
    if s.startswith("staff"):  return "Staff"
    return None

def to_coarse_and_pgy(lvl9: str) -> Tuple[int,int,int]:
    if lvl9 == "Medical student": return COARSE_TO_IDX["Student"], -1, 0
    if lvl9 == "Fellow":          return COARSE_TO_IDX["Fellow"],  -1, 0
    if lvl9 == "Staff":           return COARSE_TO_IDX["Staff"],   -1, 0
    if lvl9 and lvl9.startswith("Resident "):
        k = int(lvl9.split("PGY")[-1])
        return COARSE_TO_IDX["Resident"], PGY_TO_IDX[f"PGY{k}"], 1
    return None, -1, 0

# =========================
# IO & Utils
# =========================
def load_items(path: str):
    with open(path, "rb") as f: obj = pickle.load(f)
    if not (isinstance(obj,(list,tuple)) and obj and isinstance(obj[0], dict)):
        raise ValueError("Pickle doit être une liste de dicts.")
    return list(obj)

def to_array_CxT(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2: raise ValueError(f"data doit être 2D, reçu: {arr.shape}")
    r,c = arr.shape
    return arr if r <= c else arr.T

# ---------- Nettoyage outliers (par canal, par trial) ----------
def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12

def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = _mad(x)
    return 0.6745 * (x - med) / mad

def _interpolate_runs(y: np.ndarray, mask: np.ndarray, max_run: int) -> np.ndarray:
    """
    Interpole linéairement les runs True de 'mask' de longueur <= max_run.
    """
    y = y.copy()
    n = len(y)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < n and mask[j]: j += 1
        run_len = j - i
        if run_len <= max_run:
            x0 = i-1
            x1 = j
            if x0 >= 0 and x1 < n:
                y[i:j] = np.linspace(y[x0], y[x1], run_len+2)[1:-1]
        i = j
    return y

def clean_trial_CxT(data_CxT: np.ndarray) -> np.ndarray:
    """
    Étapes: robust z-score -> interpolation des petits runs -> filtre médian -> clipping -> z-score
    """
    C,T = data_CxT.shape
    Y = data_CxT.astype(np.float32).copy()
    # 1) robust z + interpolation + médian canal-par-canal
    for c in range(C):
        y = Y[c]
        rz = _robust_z(y)
        out = np.abs(rz) > MAD_THRESH
        if np.any(out):
            y = _interpolate_runs(y, out, MAX_INTERP_RUN)
        if MEDIAN_WIN and MEDIAN_WIN >= 3:
            y = median_filter(y, size=MEDIAN_WIN, mode="nearest")
        # clipping doux
        m = np.mean(y); s = np.std(y) + 1e-6
        y = np.clip(y, m - CLIP_SIGMA*s, m + CLIP_SIGMA*s)
        Y[c] = y
    print(rz)
    # 2) z-score par canal (trial)
    m = Y.mean(axis=1, keepdims=True); 
    s = Y.std(axis=1, keepdims=True) + 1e-6
    Y = (Y - m) / s
    return Y

# =========================
# Sélection des features par nom
# =========================
DESIRED_FEATURES = [
    "X Position","Y Position","Z Position","Position Magnitude",
    "Distance Bipolar–Cavitron","Distance Bipolar–Scissors",
]

def select_feature_indices(ch_names: List[str], desired: List[str]) -> List[int]:
    low = [n.lower() for n in ch_names]
    out_idx = []
    for want in desired:
        w = want.lower()
        found = [i for i,n in enumerate(low) if n == w]
        if not found:
            found = [i for i,n in enumerate(low) if w in n]
        if found:
            for i in found:
                if i not in out_idx:
                    out_idx.append(i); break
        else:
            print(f"[WARN] Feature demandée introuvable: '{want}'")
    return out_idx

# =========================
# Dataset builder (fenêtrage + filtre de features + nettoyage)
# =========================
def build_windows_dataset(items, L=100, hop=100, meta_path=None, desired_features=None):
    ch_names_meta = None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path,"r",encoding="utf-8") as f:
                meta = json.load(f)
                if isinstance(meta, dict) and "metric_names" in meta:
                    ch_names_meta = meta["metric_names"]
        except Exception:
            ch_names_meta = None

    X, y9, yco, ypgy, yresmask, parts, trials, starts = [], [], [], [], [], [], [], []
    kept_idx = None
    kept_names = None

    for it in items:
        lvl9 = map_level_to_9(it.get("level"))
        if lvl9 is None or lvl9 not in LEVEL_TO_IDX: 
            continue

        data = to_array_CxT(it["data"])  # (C,T)
        C,T = data.shape

        # Sélection des features via meta
        if kept_idx is None:
            if ch_names_meta and len(ch_names_meta) == C and desired_features:
                kept_idx = select_feature_indices(ch_names_meta, desired_features)
                kept_names = [ch_names_meta[i] for i in kept_idx]
            else:
                print("[WARN] Meta 'metric_names' absente ou incompatible — aucune sélection par nom. On garde tous les canaux.")
                kept_idx = list(range(C))
                kept_names = [f"ch{i}" for i in range(C)]
            if not kept_idx:
                raise ValueError("Aucune des features demandées n'a été trouvée dans le meta.")
            print("→ Features gardées:", kept_names)

        data = data[kept_idx, :]
        C,T = data.shape
        if T < L: 
            continue

        # Copie brute pour le test “>=50% zéros”
        data_raw = data.copy()
        
        # Nettoyage + normalisation (par trial)
        data_clean = clean_trial_CxT(data)

        pid = str(it.get("participant","unk"))
        tid = f"{pid}__{str(it.get('trial','t0'))}"

        for start in range(0, T-L+1, hop):
            seg_raw = data_raw[:, start:start+L]
            # Test 50% de zéros (sur données brutes sélectionnées, pas nettoyées)
            zero_ratio = float((seg_raw == 0).sum()) / float(seg_raw.size)
            if zero_ratio >= ZERO_RATIO_THRESHOLD:
                # on rejette la fenêtre
                continue

            seg = data_clean[:, start:start+L]
            X.append(seg.T.astype(np.float32))
            y9.append(LEVEL_TO_IDX[lvl9])
            co, pgy, rmask = to_coarse_and_pgy(lvl9)
            yco.append(co); ypgy.append(pgy); yresmask.append(rmask)
            parts.append(pid); trials.append(tid); starts.append(start)

    if not X: raise ValueError("Aucune fenêtre construite (vérifie L/HOP, features et filtre 50% zéros).")
    X = np.stack(X,0)
    y9 = np.asarray(y9, np.int32)
    yco = np.asarray(yco, np.int32)
    ypgy = np.asarray(ypgy, np.int32)
    yresmask = np.asarray(yresmask, np.int32)
    parts = np.asarray(parts, object)
    trials= np.asarray(trials, object)
    starts= np.asarray(starts, np.int32)

    # Sauvegarde des features retenues
    map_path = os.path.join(OUT_DIR, "metrics_used_reduced.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("Index\tMetric name\n")
        for i, nm in enumerate(kept_names):
            f.write(f"{i}\t{nm}\n")
    print(f"📝 Liste des métriques utilisées sauvegardée: {map_path}")

    print(f"Fenêtres = {len(X)} | L={SEQ_LEN} | C={X.shape[2]} | classes=9")
    return X,y9,yco,ypgy,yresmask,parts,trials,starts, kept_names

def to_categorical_int(y, n):
    Y = np.zeros((len(y), n), dtype=np.float32)
    valid = (y >= 0) & (y < n)
    Y[np.where(valid)[0], y[valid]] = 1.0
    return Y

# =========================
# Équilibrage (train only)
# =========================
def oversample_by_y9(X, y9, Yco, Ypgy, mask_res, parts, trials, starts, seed=42):
    rng = np.random.default_rng(seed)
    idx_by_c = {c: np.where(y9==c)[0] for c in range(9)}
    sizes = {c: len(v) for c,v in idx_by_c.items() if len(v)>0}
    if not sizes:
        return X,y9,Yco,Ypgy,mask_res,parts,trials,starts
    max_n = max(sizes.values())

    new_indices = []
    for c, idxs in idx_by_c.items():
        if len(idxs)==0: continue
        need = max_n - len(idxs)
        if need > 0:
            add = rng.choice(idxs, size=need, replace=True)
            idxs = np.concatenate([idxs, add])
        new_indices.extend(list(idxs))

    new_indices = np.array(new_indices, dtype=np.int64)
    rng.shuffle(new_indices)
    return (X[new_indices], y9[new_indices],
            Yco[new_indices], Ypgy[new_indices],
            mask_res[new_indices], parts[new_indices],
            trials[new_indices], starts[new_indices])

# =========================
# Modèle (LSTM + Transformer)
# =========================
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=rate)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate); self.drop2 = layers.Dropout(rate)
    def call(self, x, training=None):
        a = self.att(x,x,training=training)
        x = self.norm1(x + self.drop1(a, training=training))
        y = self.ffn(x)
        x = self.norm2(x + self.drop2(y, training=training))
        return x

def tfa_adamw(lr, weight_decay=1e-4, clipnorm=None):
    try:
        import tensorflow_addons as tfa
        opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=lr)
    if clipnorm: opt.clipnorm = clipnorm
    return opt

def build_backbone(seq_len, n_features):
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.Dense(EMBED_DIM, name="embed")(inp)
    x = layers.LayerNormalization(epsilon=1e-6, name="embed_ln")(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True), name="bilstm")(x)
    for i in range(N_BLOCKS):
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, rate=DROPOUT, name=f"tr_block{i+1}")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.25)(x)
    return keras.Model(inp, x, name="backbone")

def build_multitask(seq_len, n_features, n_coarse=4, n_fine=6):
    bb = build_backbone(seq_len, n_features)
    out_coarse = layers.Dense(n_coarse, activation="softmax", name="coarse")(bb.output)
    out_fine   = layers.Dense(n_fine,   activation="softmax", name="fine")(bb.output)
    model = keras.Model(bb.input, {"coarse": out_coarse, "fine": out_fine}, name="mtl")
    opt = tfa_adamw(BASE_LR, weight_decay=WEIGHT_DECAY, clipnorm=CLIP_NORM)
    model.compile(
        optimizer=opt,
        loss={"coarse": keras.losses.CategoricalCrossentropy(), "fine": keras.losses.CategoricalCrossentropy()},
        loss_weights={"coarse": 1.0, "fine": 1.0},
        metrics={"coarse":["accuracy"], "fine":["accuracy"]}
    )
    return model

# =========================
# Métriques & plots
# =========================
def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t,p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t,p] += 1
    return cm

def balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    cm = confusion_matrix_np(y_true, y_pred, n_classes)
    recalls = []
    for c in range(n_classes):
        tp = cm[c,c]; fn = cm[c,:].sum() - tp
        denom = tp + fn
        if denom > 0:
            recalls.append(tp/denom)
    if not recalls: return 0.0
    return float(np.mean(recalls))

def classification_report_simple(y_true, y_pred, class_names):
    cm = confusion_matrix_np(y_true, y_pred, len(class_names))
    lines = ["class\tprecision\trecall\tf1\tsupport"]
    for c, name in enumerate(class_names):
        tp = cm[c,c]; fp = cm[:,c].sum()-tp; fn = cm[c,:].sum()-tp; sup = cm[c,:].sum()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        lines.append(f"{name}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{sup}")
    acc = (y_true==y_pred).mean()
    bacc= balanced_accuracy_np(y_true,y_pred,len(class_names))
    lines.append(f"\naccuracy\t{acc:.4f}")
    lines.append(f"balanced_accuracy\t{bacc:.4f}")
    return "\n".join(lines)

def reconstruct_level9_from_heads(p_coarse, p_fine):
    N = p_coarse.shape[0]
    P9 = np.zeros((N, 9), dtype=np.float32)
    P9[:, 0] = p_coarse[:, COARSE_TO_IDX["Student"]]     # Medical student
    pres = p_coarse[:, COARSE_TO_IDX["Resident"]][:, None]
    P9[:, 1:7] = pres * p_fine                           # PGY1..6
    P9[:, 7] = p_coarse[:, COARSE_TO_IDX["Fellow"]]      # Fellow
    P9[:, 8] = p_coarse[:, COARSE_TO_IDX["Staff"]]       # Staff
    s = P9.sum(axis=1, keepdims=True) + 1e-8
    return P9 / s

def predict_heads(model, X):
    out = model.predict(X, verbose=0)
    if isinstance(out, dict):
        return out["coarse"], out["fine"]
    if isinstance(out, (list, tuple)) and len(out) == 2:
        return out[0], out[1]
    raise ValueError(f"Sortie inattendue de model.predict: type={type(out)}")

def plot_confusion(cf, labels, outpath, title="Confusion"):
    plt.figure(figsize=(8,6)); plt.imshow(cf, interpolation="nearest")
    plt.title(title); plt.xlabel("Prédit"); plt.ylabel("Vrai")
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right"); plt.yticks(ticks, labels)
    for i in range(cf.shape[0]):
        for j in range(cf.shape[1]):
            plt.text(j,i,str(cf[i,j]),ha="center",va="center",fontsize=8)
    plt.colorbar(); plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def plot_heatmap_individuals(pred_counts, row_labels, col_labels, outpath, title):
    plt.figure(figsize=(10, max(6, 0.25*len(row_labels))))
    plt.imshow(pred_counts, interpolation="nearest", aspect="auto")
    plt.title(title); plt.xlabel("Classe prédite (9-niveaux)"); plt.ylabel("Individu")
    xt = np.arange(len(col_labels))
    yt = np.arange(len(row_labels))
    plt.xticks(xt, col_labels, rotation=45, ha="right")
    plt.yticks(yt, row_labels)
    plt.colorbar(); plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

# =========================
# Main (KFold participants)
# =========================
def main():
    print(f"📁 Lecture: {DATA_PATH}")
    items = load_items(DATA_PATH)

    (X_raw, y9, yco, ypgy, yresmask, part_ids, trial_ids, start_idx,
     kept_names) = build_windows_dataset(
        items, L=SEQ_LEN, hop=HOP, meta_path=META_PATH, desired_features=DESIRED_FEATURES
    )
    N,L,C = X_raw.shape
    print(f"→ Features finales (C={C}): {kept_names}")

    # One-hot (fine: non-résidents -> vecteur 0, masqué par sample_weight)
    Yco  = to_categorical_int(yco, 4)
    Ypgy = to_categorical_int(ypgy, 6)

    # Uniques participants présents (peut être <47 si certains filtrés)
    uniq_parts_all = np.array(sorted(np.unique(part_ids)))
    print(f"Participants uniques (après filtrage): {len(uniq_parts_all)}")

    # Index des fenêtres par participant
    idx_by_part = {p: np.where(part_ids==p)[0] for p in uniq_parts_all}
    # Retirer les participants sans fenêtres (par sécurité)
    parts_with_data = np.array([p for p in uniq_parts_all if len(idx_by_part[p]) > 0])

    # KFold (5 splits) sur les participants — tirage aléatoire
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(kf.split(parts_with_data))
    print(f"CV: {len(folds)} folds (participants répartis aléatoirement).")

    # OOF collectors
    oof_pred9 = np.zeros(N, np.int32)
    oof_prob9 = np.zeros((N,9), np.float32)
    fold_id   = np.zeros(N, np.int32)

    for fold_idx, (tr_idx_p, va_idx_p) in enumerate(folds, 1):
        tr_parts = parts_with_data[tr_idx_p]
        va_parts = parts_with_data[va_idx_p]

        va = np.concatenate([idx_by_part[p] for p in va_parts])
        tr = np.setdiff1d(np.arange(N), va)

        print(f"\n===== KFold {fold_idx}/5 — val participants: {len(va_parts)} | train participants: {len(tr_parts)} =====")

        Xtr, Xva = X_raw[tr], X_raw[va]
        y9tr, y9va = y9[tr], y9[va]
        Yco_tr, Yco_va = Yco[tr], Yco[va]
        Ypgy_tr, Ypgy_va = Ypgy[tr], Ypgy[va]
        mask_tr, mask_va = yresmask[tr], yresmask[va]

        # Équilibrage TRAIN par y9
        (Xtr_bal, y9tr_bal, Yco_tr_bal, Ypgy_tr_bal,
         mask_tr_bal, parts_tr_bal, _, _) = oversample_by_y9(
            Xtr, y9tr, Yco_tr, Ypgy_tr, mask_tr, part_ids[tr], trial_ids[tr], start_idx[tr], seed=SEED+fold_idx
        )

        # Modèle
        model = build_multitask(SEQ_LEN, C, n_coarse=4, n_fine=6)
        ckpt_path = os.path.join(OUT_DIR, f"best_fold_{fold_idx}.weights.h5")

        class ValBalancedAccMTL(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                p_coarse, p_fine = predict_heads(self.model, Xva)
                P9 = reconstruct_level9_from_heads(p_coarse, p_fine)
                yhat9 = np.argmax(P9, axis=1)
                bacc9 = balanced_accuracy_np(y9va, yhat9, 9)
                logs = logs or {}
                logs["val_balanced_acc9"] = bacc9
                print(f"  -> val_balanced_acc9 = {bacc9:.4f}")

        val_bacc_cb = ValBalancedAccMTL()
        rlrop = keras.callbacks.ReduceLROnPlateau(
            monitor="val_balanced_acc9", mode="max", factor=0.5, patience=4, verbose=1
        )

        # Masque PGY train/val
        sw_coarse_tr = np.ones(len(Xtr_bal), dtype=np.float32)
        sw_fine_tr   = mask_tr_bal.astype(np.float32)
        sw_coarse_va = np.ones(len(Xva), dtype=np.float32)
        sw_fine_va   = mask_va.astype(np.float32)

        model.fit(
            Xtr_bal, {"coarse": Yco_tr_bal, "fine": Ypgy_tr_bal},
            validation_data=(Xva, {"coarse": Yco_va, "fine": Ypgy_va},
                             {"coarse": sw_coarse_va, "fine": sw_fine_va}),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
            callbacks=[val_bacc_cb, rlrop],
            sample_weight={"coarse": sw_coarse_tr, "fine": sw_fine_tr}
        )

        # Prédictions validation
        p_coarse, p_fine = predict_heads(model, Xva)
        P9_va = reconstruct_level9_from_heads(p_coarse, p_fine)
        yhat9_va = np.argmax(P9_va, axis=1)

        oof_pred9[va] = yhat9_va
        oof_prob9[va] = P9_va
        fold_id[va]   = fold_idx

        acc = (y9va == yhat9_va).mean()
        bacc= balanced_accuracy_np(y9va, yhat9_va, 9)
        print(f"[Fold {fold_idx}] acc9={acc:.4f} | bal_acc9={bacc:.4f}")

    # ===== Résultats OOF (fenêtres) =====
    rep_txt = classification_report_simple(y9, oof_pred9, LEVEL9_ORDER)
    with open(os.path.join(OUT_DIR,"report_windows_oof.txt"),"w",encoding="utf-8") as f: f.write(rep_txt)
    print("\n=== OOF fenêtres (9 classes) ===\n"+rep_txt)

    cf_win = confusion_matrix_np(y9, oof_pred9, 9)
    plot_confusion(cf_win, LEVEL9_ORDER, os.path.join(OUT_DIR,"confusion_windows_oof.png"),
                   title="Matrice de confusion — fenêtres (OOF, 9 classes)")

    # ===== “Confusion par individu” (fenêtres : individu × classes9) =====
    # Pour chaque individu, compter les classes prédites sur ses fenêtres OOF
    uniq_parts_present = np.array(sorted(np.unique(part_ids)))
    pred_counts = np.zeros((len(uniq_parts_present), 9), dtype=np.int64)
    for i, pid in enumerate(uniq_parts_present):
        idx = np.where(part_ids == pid)[0]
        if len(idx) > 0:
            counts = np.bincount(oof_pred9[idx], minlength=9)
            pred_counts[i,:] = counts
    # heatmap individus × classes
    plot_heatmap_individuals(
        pred_counts,
        row_labels=list(map(str, uniq_parts_present)),
        col_labels=LEVEL9_ORDER,
        outpath=os.path.join(OUT_DIR,"confusion_participants_oof.png"),
        title="Distribution des prédictions par individu (fenêtres)"
    )

    # ===== Agrégation par participant (majorité) =====
    def majority_aggregate(y_pred: np.ndarray, ids: np.ndarray) -> Dict[str,int]:
        out = {}
        for ent in np.unique(ids):
            cls, cnt = np.unique(y_pred[ids==ent], return_counts=True)
            out[ent] = int(cls[np.argmax(cnt)])
        return out
    def mode_true_by_entity(y_true: np.ndarray, ids: np.ndarray) -> Dict[str,int]:
        out = {}
        for ent in np.unique(ids):
            cls, cnt = np.unique(y_true[ids==ent], return_counts=True)
            out[ent] = int(cls[np.argmax(cnt)])
        return out
    def dict_to_arrays(d_pred: Dict[str,int], d_true: Dict[str,int]):
        ents = sorted(set(d_pred.keys()) & set(d_true.keys()))
        yhat = np.array([d_pred[e] for e in ents], dtype=np.int32)
        ytru = np.array([d_true[e] for e in ents], dtype=np.int32)
        return ents, ytru, yhat

    d_pred_part = majority_aggregate(oof_pred9, part_ids)
    d_true_part = mode_true_by_entity(y9, part_ids)
    ents, ytrue_part, ypred_part = dict_to_arrays(d_pred_part, d_true_part)

    rep_part = classification_report_simple(ytrue_part, ypred_part, LEVEL9_ORDER)
    with open(os.path.join(OUT_DIR,"report_participants_oof.txt"),"w",encoding="utf-8") as f: f.write(rep_part)
    print("\n=== OOF participants (majorité, 9 classes) ===\n"+rep_part)

    cf_part = confusion_matrix_np(ytrue_part, ypred_part, 9)
    plot_confusion(cf_part, LEVEL9_ORDER, os.path.join(OUT_DIR,"confusion_participants_majority_oof.png"),
                   title="Matrice de confusion — participants (agrégés, 9 classes)")

    # Exports CSV (fenêtres)
    win_csv = os.path.join(OUT_DIR,"pred_windows_oof.csv")
    with open(win_csv,"w",encoding="utf-8") as f:
        header=["fold","participant","trial","start","true9","pred9"]+[f"p_{c}" for c in LEVEL9_ORDER]
        f.write(",".join(header)+"\n")
        for i in range(len(y9)):
            row=[str(int(fold_id[i])), str(part_ids[i]), str(trial_ids[i]), str(int(start_idx[i])),
                 LEVEL9_ORDER[int(y9[i])], LEVEL9_ORDER[int(oof_pred9[i])]] + [f"{float(oof_prob9[i,j]):.6f}" for j in range(9)]
            f.write(",".join(row)+"\n")

    # Exports CSV (participants agrégés)
    part_csv = os.path.join(OUT_DIR,"pred_participants_oof.csv")
    with open(part_csv,"w",encoding="utf-8") as f:
        f.write("participant,true9,pred9\n")
        for ent,yt,yp in zip(ents,ytrue_part,ypred_part):
            f.write(f"{ent},{LEVEL9_ORDER[int(yt)]},{LEVEL9_ORDER[int(yp)]}\n")

    print("\n✅ Terminé (KFold=5 participants, filtre 50% zéros, features pos+dist, confusions par niveau et par individu).")

if __name__ == "__main__":
    main()
