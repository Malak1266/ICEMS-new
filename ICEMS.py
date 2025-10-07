# -*- coding: utf-8 -*-
"""
CV 5-fold by TRIALS — LSTM+Transformer — Full metrics — Train-only normalization — Test holdout

Key points (per user spec):
- Windows: L=100, hop=100 (10s @ 10Hz), remove windows with >=50% zeros (raw pre-clean)
- Use ALL metrics produced by generate_data.py (X,Y,Z, PosMag, Velocity, Acceleration, Jerk, Distances)
- CV split by TRIALS (Group = trial id) so train/val never share the same trial
- Ensure level coverage in train via StratifiedKFold on trials (using the most frequent level of each trial)
- Normalization: compute mean/std on TRAIN ONLY, then apply to VAL and TEST
- Final holdout TEST set: ~15% of trials completely unseen during CV; final model trained on remaining trials
- Outputs:
    * pred_windows_oof.csv (fold, participant, trial, start, true label 9, pred label 9, probs per class)
    * confusion_windows_oof.png  (windows OOF)
    * confusion_participants_oof.png (heatmap: participants × predicted classes counts)
    * confusion_participants_majority_oof.png (participants aggregated by majority to 9-level)
    * test_pred_windows.csv, test_pred_participants.csv
    * confusion_windows_test.png, confusion_participants_majority_test.png
"""

import os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle as sk_shuffle

# ------------- Config -------------
DATA_PATH = "./data/final_from_full_A.pkl"   # update if needed
META_PATH = os.path.splitext(DATA_PATH)[0] + "_meta.json"
OUT_DIR   = "./runs_cv_by_trials_fullmetrics"

SEQ_LEN    = 100
HOP        = SEQ_LEN
BATCH_SIZE = 64
EPOCHS     = 60
SEED       = 42

# Model
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

# Cleaning
MAD_THRESH         = 6.0
MAX_INTERP_RUN     = 20
MEDIAN_WIN         = 3
CLIP_SIGMA         = 5.0

ZERO_RATIO_THRESHOLD = 0.50  # reject windows with >=50% zeros

TEST_TRIAL_FRACTION = 0.15   # portion of trials reserved as final test (never seen in CV)

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- Labels (9-level layout) -------------
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

# ------------- IO & Utilities -------------
def load_items(path: str):
    with open(path, "rb") as f: obj = pickle.load(f)
    if not (isinstance(obj,(list,tuple)) and obj and isinstance(obj[0], dict)):
        raise ValueError("Pickle must be a list of dicts (entries).")
    return list(obj)

def to_array_CxT(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2: raise ValueError(f"data must be 2D, got: {arr.shape}")
    r,c = arr.shape
    return arr if r <= c else arr.T

def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12

def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = _mad(x)
    return 0.6745 * (x - med) / mad

def _interpolate_runs(y: np.ndarray, mask: np.ndarray, max_run: int) -> np.ndarray:
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

def clean_trial_CxT_no_norm(data_CxT: np.ndarray) -> np.ndarray:
    """
    Clean per channel per trial WITHOUT per-trial normalization (to avoid leakage).
    Steps: robust z to detect spikes -> small-run interpolation -> median filter -> soft clipping.
    """
    C,T = data_CxT.shape
    Y = data_CxT.astype(np.float32).copy()
    for c in range(C):
        y = Y[c]
        rz = _robust_z(y)
        out = np.abs(rz) > MAD_THRESH
        if np.any(out):
            y = _interpolate_runs(y, out, MAX_INTERP_RUN)
        if MEDIAN_WIN and MEDIAN_WIN >= 3:
            from scipy.ndimage import median_filter
            y = median_filter(y, size=MEDIAN_WIN, mode="nearest")
        m = np.mean(y); s = np.std(y) + 1e-6
        y = np.clip(y, m - CLIP_SIGMA*s, m + CLIP_SIGMA*s)
        Y[c] = y
    return Y

def compute_train_norm_stats(X_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean/std per channel computed on TRAIN windows only.
    X_tr: [N, L, C]
    Returns (mean[C], std[C])
    """
    mean = X_tr.mean(axis=(0,1))
    std  = X_tr.std(axis=(0,1)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)

# ------------- Feature selection -------------
def select_feature_indices(ch_names: List[str]) -> List[int]:
    # Keep everything except the two label rows which MUST be first in the file.
    # We rely on meta["metric_names"] to match data rows.
    # metric_names includes: ["Label(Expertise)","Label(Level)", <features...>]
    # We'll drop idx 0 and 1 (labels) and keep the rest in the exact order.
    return list(range(2, len(ch_names)))

# ------------- Dataset windows builder -------------
def build_windows_dataset(items, L=100, hop=100, meta_path=None):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ch_names_meta = meta.get("metric_names", None)

    X, y9, yco, ypgy, yresmask, parts, trials, starts = [], [], [], [], [], [], [], []
    kept_idx = None
    kept_names = None

    for it in items:
        lvl_raw = it.get("level")
        lvl9 = map_level_to_9(lvl_raw)
        if lvl9 is None or lvl9 not in LEVEL_TO_IDX:
            continue

        data = to_array_CxT(it["data"])  # (C,T)
        C,T = data.shape

        if kept_idx is None:
            if ch_names_meta and len(ch_names_meta) == C:
                kept_idx = select_feature_indices(ch_names_meta)   # drop the 2 label rows
                kept_names = [ch_names_meta[i] for i in kept_idx]
            else:
                # fallback: assume labels are the first two rows
                kept_idx = list(range(2, C))
                kept_names = [f"feat_{i}" for i in kept_idx]
            if not kept_idx:
                raise ValueError("No features found to keep.")
            print("→ Features used:", kept_names)

        # Select features only (drop label rows)
        feats = data[kept_idx, :]
        C2,T2 = feats.shape
        if T2 < L:
            continue

        # Keep a raw copy to compute zero ratio
        feats_raw = feats.copy()
        feats_cln = clean_trial_CxT_no_norm(feats)

        pid = str(it.get("participant","unk"))
        tid = f"{pid}__{str(it.get('trial','t0'))}"

        for start in range(0, T2-L+1, hop):
            seg_raw = feats_raw[:, start:start+L]
            zero_ratio = float((seg_raw == 0).sum()) / float(seg_raw.size)
            if zero_ratio >= ZERO_RATIO_THRESHOLD:
                continue  # drop window with too many zeros

            seg = feats_cln[:, start:start+L]
            X.append(seg.T.astype(np.float32))  # [L,C]
            y9.append(LEVEL_TO_IDX[lvl9])
            co, pgy, rmask = to_coarse_and_pgy(lvl9)
            yco.append(co); ypgy.append(pgy); yresmask.append(rmask)
            parts.append(pid); trials.append(tid); starts.append(start)

    if not X:
        raise ValueError("No windows built. Check filters and L/HOP.")
    X = np.stack(X, 0)
    y9 = np.asarray(y9, np.int32)
    yco = np.asarray(yco, np.int32)
    ypgy = np.asarray(ypgy, np.int32)
    yresmask = np.asarray(yresmask, np.int32)
    parts = np.asarray(parts, object)
    trials= np.asarray(trials, object)
    starts= np.asarray(starts, np.int32)

    # Save which metrics were used (for traceability)
    map_path = os.path.join(OUT_DIR, "metrics_used_full.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("Index\tMetric name\n")
        for i, nm in enumerate(kept_names):
            f.write(f"{i}\t{nm}\n")

    print(f"Windows built = {len(X)} | L={L} | C={X.shape[2]} | classes=9")
    return X,y9,yco,ypgy,yresmask,parts,trials,starts, kept_names

def to_categorical_int(y, n):
    Y = np.zeros((len(y), n), dtype=np.float32)
    valid = (y >= 0) & (y < n)
    Y[np.where(valid)[0], y[valid]] = 1.0
    return Y

# ------------- Model -------------
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

# ------------- Metrics/plots -------------
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
    raise ValueError(f"Unexpected model.predict output: {type(out)}")

def plot_confusion(cf, labels, outpath, title="Confusion"):
    plt.figure(figsize=(8,6)); plt.imshow(cf, interpolation="nearest")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right"); plt.yticks(ticks, labels)
    for i in range(cf.shape[0]):
        for j in range(cf.shape[1]):
            plt.text(j,i,str(cf[i,j]),ha="center",va="center",fontsize=8)
    plt.colorbar(); plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def plot_heatmap_individuals(pred_counts, row_labels, col_labels, outpath, title):
    plt.figure(figsize=(10, max(6, 0.25*len(row_labels))))
    plt.imshow(pred_counts, interpolation="nearest", aspect="auto")
    plt.title(title); plt.xlabel("Pred class (9)"); plt.ylabel("Participant")
    xt = np.arange(len(col_labels))
    yt = np.arange(len(row_labels))
    plt.xticks(xt, col_labels, rotation=45, ha="right")
    plt.yticks(yt, row_labels)
    plt.colorbar(); plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

# ------------- Trial-level stratification -------------
def stratified_trial_splits(trial_ids: np.ndarray, y9: np.ndarray, n_splits: int, seed: int):
    """
    Stratify by the majority 9-level label per TRIAL.
    Returns a list of (train_idx, val_idx) over WINDOWS indices.
    """
    uniq_trials = np.array(sorted(np.unique(trial_ids)))
    # Majority label per trial (from windows y9)
    trial_to_idx = {t:i for i,t in enumerate(uniq_trials)}
    trial_labels = np.zeros(len(uniq_trials), np.int32)
    for t in uniq_trials:
        idx = np.where(trial_ids == t)[0]
        cls, cnt = np.unique(y9[idx], return_counts=True)
        trial_labels[trial_to_idx[t]] = int(cls[np.argmax(cnt)])

    # Create stratified folds on trials
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for tr_trials_idx, va_trials_idx in skf.split(uniq_trials, trial_labels):
        tr_trials = uniq_trials[tr_trials_idx]
        va_trials = uniq_trials[va_trials_idx]
        tr = np.where(np.isin(trial_ids, tr_trials))[0]
        va = np.where(np.isin(trial_ids, va_trials))[0]
        splits.append((tr,va))
    return splits

# ------------- Majority helpers -------------
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

# ------------- Main -------------
def main():
    print(f"📁 Loading: {DATA_PATH}")
    items = load_items(DATA_PATH)
    (X_raw, y9, yco, ypgy, yresmask, part_ids, trial_ids, start_idx,
     kept_names) = build_windows_dataset(items, L=SEQ_LEN, hop=HOP, meta_path=META_PATH)
    N,L,C = X_raw.shape
    print(f"→ Final features (C={C}): {kept_names}")

    # One-hot targets
    def to_cat(y, n):
        Y = np.zeros((len(y),n), np.float32); Y[np.arange(len(y)), y] = 1.0; return Y
    Yco  = to_cat(np.asarray([COARSE_TO_IDX["Resident"] if "Resident" in LEVEL9_ORDER[i] else
                              COARSE_TO_IDX["Student"] if i==0 else
                              COARSE_TO_IDX["Fellow"]  if i==7 else
                              COARSE_TO_IDX["Staff"]   if i==8 else 0 for i in y9], dtype=np.int32), 4)
    # more explicit mapping using existing helpers
    Yco = np.zeros((len(y9),4), np.float32)
    Ypgy= np.zeros((len(y9),6), np.float32)
    for i,(lvl_idx) in enumerate(y9):
        name = LEVEL9_ORDER[int(lvl_idx)]
        co,pgy, rmask = to_coarse_and_pgy(name)
        if co is not None and 0<=co<4: Yco[i, co] = 1.0
        if 0<=pgy<6: Ypgy[i, pgy] = 1.0
    mask_res = np.array([1 if LEVEL9_ORDER[int(k)].startswith("Resident ") else 0 for k in y9], np.int32)

    # ---------------- Test holdout (by TRIALS) ----------------
    uniq_trials = np.array(sorted(np.unique(trial_ids)))
    rng = np.random.default_rng(SEED)
    test_trials = rng.choice(uniq_trials, size=max(1, int(round(len(uniq_trials)*TEST_TRIAL_FRACTION))), replace=False)
    not_test_trials = uniq_trials[~np.isin(uniq_trials, test_trials)]

    test_idx = np.where(np.isin(trial_ids, test_trials))[0]
    cv_idx   = np.where(np.isin(trial_ids, not_test_trials))[0]

    # ---------------- CV (stratified by TRIAL majority) ----------------
    splits = stratified_trial_splits(trial_ids[cv_idx], y9[cv_idx], n_splits=5, seed=SEED)

    oof_pred9 = np.zeros(N, np.int32)
    oof_prob9 = np.zeros((N,9), np.float32)
    fold_id   = np.zeros(N, np.int32)

    for fold_idx, (tr_rel, va_rel) in enumerate(splits, 1):
        tr = cv_idx[tr_rel]; va = cv_idx[va_rel]

        Xtr_raw, Xva_raw = X_raw[tr], X_raw[va]
        y9tr, y9va = y9[tr], y9[va]
        # rebuild heads
        Yco_tr = np.zeros((len(tr),4), np.float32)
        Ypgy_tr= np.zeros((len(tr),6), np.float32)
        mask_tr= np.zeros((len(tr),), np.float32)
        for i,(idx) in enumerate(tr):
            name = LEVEL9_ORDER[int(y9[idx])]
            co,pgy,rm = to_coarse_and_pgy(name); mask_tr[i]=rm
            if co is not None and 0<=co<4: Yco_tr[i,co]=1.0
            if 0<=pgy<6: Ypgy_tr[i,pgy]=1.0

        Yco_va = np.zeros((len(va),4), np.float32)
        Ypgy_va= np.zeros((len(va),6), np.float32)
        mask_va= np.zeros((len(va),), np.float32)
        for i,(idx) in enumerate(va):
            name = LEVEL9_ORDER[int(y9[idx])]
            co,pgy,rm = to_coarse_and_pgy(name); mask_va[i]=rm
            if co is not None and 0<=co<4: Yco_va[i,co]=1.0
            if 0<=pgy<6: Ypgy_va[i,pgy]=1.0

        # Train-only normalization
        mean_c, std_c = compute_train_norm_stats(Xtr_raw)
        Xtr = apply_norm(Xtr_raw, mean_c, std_c)
        Xva = apply_norm(Xva_raw, mean_c, std_c)

        # Balance train by y9 (simple oversampling)
        idx_by_c = {c: np.where(y9tr==c)[0] for c in range(9)}
        sizes = {c: len(v) for c,v in idx_by_c.items() if len(v)>0}
        max_n = max(sizes.values()) if sizes else 0
        rng_local = np.random.default_rng(SEED+fold_idx)
        new_tr_idx = []
        for c, idxs in idx_by_c.items():
            if len(idxs)==0: continue
            need = max_n - len(idxs)
            if need > 0:
                add = rng_local.choice(idxs, size=need, replace=True)
                idxs = np.concatenate([idxs, add])
            new_tr_idx.extend(list(idxs))
        new_tr_idx = np.array(new_tr_idx, np.int64)
        rng_local.shuffle(new_tr_idx)

        Xtr = Xtr[new_tr_idx]
        y9tr_bal = y9tr[new_tr_idx]
        Yco_tr = Yco_tr[new_tr_idx]
        Ypgy_tr= Ypgy_tr[new_tr_idx]
        mask_tr= mask_tr[new_tr_idx]

        # Model
        model = build_multitask(SEQ_LEN, Xtr.shape[2], n_coarse=4, n_fine=6)

        class ValBalancedAcc9(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                p_coarse, p_fine = predict_heads(self.model, Xva)
                P9 = reconstruct_level9_from_heads(p_coarse, p_fine)
                yhat9 = np.argmax(P9, axis=1)
                bacc9 = balanced_accuracy_np(y9va, yhat9, 9)
                print(f"  -> val_balanced_acc9 = {bacc9:.4f}")

        rlrop = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                                  factor=0.5, patience=4, verbose=1)

        sw_coarse_tr = np.ones(len(Xtr), dtype=np.float32)
        sw_fine_tr   = mask_tr.astype(np.float32)
        sw_coarse_va = np.ones(len(Xva), dtype=np.float32)
        sw_fine_va   = mask_va.astype(np.float32)

        model.fit(
            Xtr, {"coarse": Yco_tr, "fine": Ypgy_tr},
            validation_data=(Xva, {"coarse": Yco_va, "fine": Ypgy_va},
                             {"coarse": sw_coarse_va, "fine": sw_fine_va}),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
            callbacks=[ValBalancedAcc9(), rlrop],
            sample_weight={"coarse": sw_coarse_tr, "fine": sw_fine_tr}
        )

        p_coarse, p_fine = predict_heads(model, Xva)
        P9_va = reconstruct_level9_from_heads(p_coarse, p_fine)
        yhat9_va = np.argmax(P9_va, axis=1)

        oof_pred9[va] = yhat9_va
        oof_prob9[va] = P9_va
        fold_id[va]   = fold_idx

        acc = (y9va == yhat9_va).mean()
        bacc= balanced_accuracy_np(y9va, yhat9_va, 9)
        print(f"[Fold {fold_idx}] acc9={acc:.4f} | bal_acc9={bacc:.4f}")

    # ===== OOF Results (windows) =====
    def save_report_and_plots(tag, y_true, y_pred, ids_for_heatmap=None):
        rep_txt = classification_report_simple(y_true, y_pred, LEVEL9_ORDER)
        with open(os.path.join(OUT_DIR,f"report_windows_{tag}.txt"),"w",encoding="utf-8") as f: f.write(rep_txt)
        print(f"\n=== {tag.upper()} windows (9 classes) ===\n"+rep_txt)
        cf = confusion_matrix_np(y_true, y_pred, 9)
        plot_confusion(cf, LEVEL9_ORDER, os.path.join(OUT_DIR,f"confusion_windows_{tag}.png"),
                       title=f"Confusion — windows ({tag}, 9 classes)")
        if ids_for_heatmap is not None:
            uniq_parts_present = np.array(sorted(np.unique(ids_for_heatmap)))
            pred_counts = np.zeros((len(uniq_parts_present), 9), dtype=np.int64)
            for i, pid in enumerate(uniq_parts_present):
                idx = np.where(ids_for_heatmap == pid)[0]
                if len(idx) > 0:
                    counts = np.bincount(y_pred[idx], minlength=9)
                    pred_counts[i,:] = counts
            plot_heatmap_individuals(
                pred_counts,
                row_labels=list(map(str, uniq_parts_present)),
                col_labels=LEVEL9_ORDER,
                outpath=os.path.join(OUT_DIR,f"confusion_participants_{tag}.png"),
                title=f"Distribution of predictions per participant ({tag})"
            )

    save_report_and_plots("oof", y9, oof_pred9, ids_for_heatmap=part_ids)

    # CSV (windows OOF with probs)
    win_csv = os.path.join(OUT_DIR,"pred_windows_oof.csv")
    with open(win_csv,"w",encoding="utf-8") as f:
        header=["fold","participant","trial","start","true9","pred9"]+[f"p_{c}" for c in LEVEL9_ORDER]
        f.write(",".join(header)+"\n")
        for i in range(len(y9)):
            row=[str(int(fold_id[i])), str(part_ids[i]), str(trial_ids[i]), str(int(start_idx[i])),
                 LEVEL9_ORDER[int(y9[i])], LEVEL9_ORDER[int(oof_pred9[i])]] + [f"{float(oof_prob9[i,j]):.6f}" for j in range(9)]
            f.write(",".join(row)+"\n")

    # Participant-level (majority) from OOF
    d_pred_part = majority_aggregate(oof_pred9, part_ids)
    d_true_part = mode_true_by_entity(y9, part_ids)
    ents, ytrue_part, ypred_part = dict_to_arrays(d_pred_part, d_true_part)
    rep_part = classification_report_simple(ytrue_part, ypred_part, LEVEL9_ORDER)
    with open(os.path.join(OUT_DIR,"report_participants_oof.txt"),"w",encoding="utf-8") as f: f.write(rep_part)
    print("\n=== OOF participants (majority, 9 classes) ===\n"+rep_part)
    cf_part = confusion_matrix_np(ytrue_part, ypred_part, 9)
    plot_confusion(cf_part, LEVEL9_ORDER, os.path.join(OUT_DIR,"confusion_participants_majority_oof.png"),
                   title="Confusion — participants (majority, OOF)")

    # ---------------- Final TEST evaluation ----------------
    # Train on all non-test windows (cv_idx), normalize on that, eval on test_idx
    if len(test_idx) > 0:
        Xtr_all_raw = X_raw[cv_idx]; y9_tr_all = y9[cv_idx]
        Yco_all = np.zeros((len(cv_idx),4), np.float32)
        Ypgy_all= np.zeros((len(cv_idx),6), np.float32)
        mask_all= np.zeros((len(cv_idx),), np.float32)
        for i,(idx) in enumerate(cv_idx):
            name = LEVEL9_ORDER[int(y9[idx])]
            co,pgy,rm = to_coarse_and_pgy(name); mask_all[i]=rm
            if co is not None and 0<=co<4: Yco_all[i,co]=1.0
            if 0<=pgy<6: Ypgy_all[i,pgy]=1.0

        mean_c, std_c = compute_train_norm_stats(Xtr_all_raw)
        Xtr_all = apply_norm(Xtr_all_raw, mean_c, std_c)
        Xte = apply_norm(X_raw[test_idx], mean_c, std_c)
        y9_te = y9[test_idx]

        model = build_multitask(SEQ_LEN, Xtr_all.shape[2], n_coarse=4, n_fine=6)
        sw_coarse = np.ones(len(Xtr_all), dtype=np.float32)
        sw_fine   = mask_all.astype(np.float32)
        model.fit(Xtr_all, {"coarse": Yco_all, "fine": Ypgy_all},
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
                  sample_weight={"coarse": sw_coarse, "fine": sw_fine})

        p_coarse, p_fine = predict_heads(model, Xte)
        P9_te = reconstruct_level9_from_heads(p_coarse, p_fine)
        yhat9_te = np.argmax(P9_te, axis=1)

        # windows report + confusion
        save_report_and_plots("test", y9_te, yhat9_te, ids_for_heatmap=part_ids[test_idx])

        # CSVs
        test_win_csv = os.path.join(OUT_DIR,"test_pred_windows.csv")
        with open(test_win_csv,"w",encoding="utf-8") as f:
            header=["participant","trial","start","true9","pred9"]+[f"p_{c}" for c in LEVEL9_ORDER]
            f.write(",".join(header)+"\n")
            for i,gi in enumerate(test_idx):
                row=[str(part_ids[gi]), str(trial_ids[gi]), str(int(start_idx[gi])),
                     LEVEL9_ORDER[int(y9[gi])], LEVEL9_ORDER[int(yhat9_te[i])]] + [f"{float(P9_te[i,j]):.6f}" for j in range(9)]
                f.write(",".join(row)+"\n")

        d_pred_part_t = majority_aggregate(yhat9_te, part_ids[test_idx])
        d_true_part_t = mode_true_by_entity(y9[test_idx], part_ids[test_idx])
        ents_t, ytrue_part_t, ypred_part_t = dict_to_arrays(d_pred_part_t, d_true_part_t)
        rep_part_t = classification_report_simple(ytrue_part_t, ypred_part_t, LEVEL9_ORDER)
        with open(os.path.join(OUT_DIR,"test_report_participants.txt"),"w",encoding="utf-8") as f: f.write(rep_part_t)
        print("\n=== TEST participants (majority, 9 classes) ===\n"+rep_part_t)
        cf_part_t = confusion_matrix_np(ytrue_part_t, ypred_part_t, 9)
        plot_confusion(cf_part_t, LEVEL9_ORDER, os.path.join(OUT_DIR,"confusion_participants_majority_test.png"),
                       title="Confusion — participants (majority, TEST)")

    print("\n✅ Done. Check outputs under:", OUT_DIR)

if __name__ == "__main__":
    main()
