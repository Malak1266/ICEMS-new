# -*- coding: utf-8 -*-
"""
Leave-One-Trial-Out (LOOCV) — LSTM+Transformer — Train-only normalization

Version multi-instruments (fenêtre unique par trial/intervalle):
- Chaque fenêtre regroupe les 3 instruments (bipolar → scissors → cavitron) concaténés
- 4 métriques par instrument: PosMag, Velocity, Acceleration, Jerk   # <<< CHANGEMENT
- + 2 canaux globaux: Distance Bipolar–Cavitron, Distance Bipolar–Scissors
- Fenêtres non chevauchées (L=100, hop=100)
- Normalisation calculée sur TRAIN uniquement, appliquée à VAL
- Sortie OOF + matrices de confusion par instrument (basées sur présence dans la fenêtre)
"""

import os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------- Config -------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "final_from_full_B.pkl")
META_PATH = os.path.splitext(DATA_PATH)[0] + "_meta.json"
OUT_DIR   = os.path.join(".", "Results_ICEMS_multi_full_LOOCV")

SEQ_LEN    = 100
HOP        = SEQ_LEN          # -> aucun chevauchement
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

# Nettoyage léger (avant normalisation)
MAD_THRESH         = 6.0
MAX_INTERP_RUN     = 20
MEDIAN_WIN         = 3
CLIP_SIGMA         = 5.0

ZERO_RATIO_THRESHOLD = 0.75  # rejeter fenêtres avec >=75% de zéros

# Instruments gérés et ordre fixe
INST_ORDER = ["bipolar", "scissors", "cavitron"]

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- CSV Grouping (agrégation par participant-trial) -------------
def group_predictions_by_trial(csv_file_path: str) -> str:
    if not os.path.exists(csv_file_path):
        print(f"Warning: File {csv_file_path} does not exist. Skipping grouping.")
        return None
    print(f"Grouping predictions from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    probability_columns = [col for col in df.columns if col.startswith('p_')]
    required_columns = ['participant', 'trial', 'true9', 'pred9'] + probability_columns
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return None
    agg = {'true9': lambda x: x.mode().iloc[0], 'pred9': lambda x: x.mode().iloc[0]}
    for col in probability_columns: agg[col] = 'mean'
    df_grouped = df.groupby(['participant', 'trial']).agg(agg).reset_index()
    for col in probability_columns: df_grouped[col] = df_grouped[col].round(6)
    out = os.path.splitext(csv_file_path)[0] + "_grouped_by_trial.csv"
    df_grouped.to_csv(out, index=False)
    print(f"Grouped data saved to: {out}")
    return out

# ------------- Labels (9 classes) -------------
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

# ------------- IO & Utils -------------
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
    n = len(y); i = 0
    while i < n:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < n and mask[j]: j += 1
        run_len = j - i
        if run_len <= max_run:
            x0 = i-1; x1 = j
            if x0 >= 0 and x1 < n:
                y[i:j] = np.linspace(y[x0], y[x1], run_len+2)[1:-1]
        i = j
    return y

def clean_trial_CxT_no_norm(data_CxT: np.ndarray) -> np.ndarray:
    """Nettoyage canal-par-canal sans normalisation (évite les fuites)."""
    from scipy.ndimage import median_filter
    C,T = data_CxT.shape
    Y = data_CxT.astype(np.float32).copy()
    for c in range(C):
        y = Y[c]
        rz = _robust_z(y)
        out = np.abs(rz) > MAD_THRESH
        if np.any(out):
            y = _interpolate_runs(y, out, MAX_INTERP_RUN)
        if MEDIAN_WIN and MEDIAN_WIN >= 3:
            y = median_filter(y, size=MEDIAN_WIN, mode="nearest")
        m = np.mean(y); s = np.std(y) + 1e-6
        y = np.clip(y, m - CLIP_SIGMA*s, m + CLIP_SIGMA*s)
        Y[c] = y
    return Y

def compute_train_norm_stats(X_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_tr.mean(axis=(0,1))
    std  = X_tr.std(axis=(0,1)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)

# ------------- Sélection des features mono-instrument -------------
# Dans le PKL "Format B", les lignes sont:
# 0: Label(Expertise), 1: Label(Level),
# 2: PosMag, 3: Velocity, 4: Acceleration, 5: Jerk,
# 6: Dist Bipolar–Cavitron, 7: Dist Bipolar–Scissors
MONO_IDX_BLOCK = [2,3,4,5]     # <<< CHANGEMENT : 4 canaux par instrument (sans X/Y/Z)
DIST_IDX       = [6,7]         # <<< CHANGEMENT : indices des distances dans les entrées mono

# ------------- Build fenêtres multi-instruments -------------
def build_windows_dataset_multi(items, L=100, hop=100):
    """
    Construit des fenêtres qui concatènent, pour un même (participant,trial):
      [bloc bipolar 4ch] + [bloc scissors 4ch] + [bloc cavitron 4ch] + [dist_bip_cav, dist_bip_sci]   # <<< CHANGEMENT

    Si un instrument n'est pas présent: bloc de zéros.
    Les distances sont prises depuis n'importe quelle entrée du trial où elles sont disponibles
    (mono avec injection OU entrées "distance-only"), masquées en amont par le générateur.
    """
    # Grouper items par (participant, trial, instrument)
    by_pti = {}
    for it in items:
        pid  = str(it.get("participant", ""))
        tr   = str(it.get("trial", ""))
        inst = str(it.get("instrument", "")).lower()
        if inst not in INST_ORDER and not inst.startswith("bipolar_to_"):
            continue
        by_pti.setdefault((pid,tr), {}).setdefault(inst, []).append(it)

    X, y9, yco, ypgy, yresmask = [], [], [], [], []
    parts, trials, starts = [], [], []
    inst_presence = []   # [N, 3] bool
    inst_combo    = []   # texte "bipolar|scissors|cavitron" (présents)
    kept_names = []

    # noms des features par instrument (4 canaux)
    for inst in INST_ORDER:
        kept_names += [f"{inst}:PosMag", f"{inst}:Velocity",
                       f"{inst}:Acceleration", f"{inst}:Jerk"]             # <<< CHANGEMENT
    kept_names += ["Distance Bipolar–Cavitron", "Distance Bipolar–Scissors"]

    def _first_array(lst_items, idxs):
        """Retourne le premier array (len=T) trouvé aux indices `idxs` dans .data"""
        for it in lst_items:
            A = to_array_CxT(it["data"])  # (rows, T)
            if A.shape[0] > max(idxs):
                return A[idxs, :]
        return None

    def _level_from_any(lst_items):
        for it in lst_items:
            lvl_raw = it.get("level")
            lvl9 = map_level_to_9(lvl_raw)
            if lvl9 is not None:
                return lvl9
        return None

    for (pid,tr), inst_map in by_pti.items():
        # rassembler matrices mono par instrument
        mono = {}
        presence = []
        for inst in INST_ORDER:
            arr = _first_array(inst_map.get(inst, []), MONO_IDX_BLOCK)  # (4, T) ou None  # <<< CHANGEMENT
            mono[inst] = arr
            presence.append(arr is not None)
        presence = np.array(presence, dtype=bool)

        # tailles → aligner sur min T des blocs présents (ou des distances si mono manquent)
        Ts = [mono[k].shape[1] for k in INST_ORDER if mono[k] is not None]
        # distances globales (peuvent venir d'entrées mono ou distance-only)
        dist_bip_cav = _first_array(inst_map.get("bipolar_to_cavitron", []), [0])  # (1,T) si distance-only
        dist_bip_sci = _first_array(inst_map.get("bipolar_to_scissors", []), [0])
        # si distances non trouvées via distance-only, tenter via entrées mono (lignes 6,7 du Format B)  # <<< CHANGEMENT
        if dist_bip_cav is None or dist_bip_sci is None:
            for inst in INST_ORDER:
                arr = _first_array(inst_map.get(inst, []), DIST_IDX)
                if arr is not None:
                    if dist_bip_cav is None: dist_bip_cav = arr[[0], :]
                    if dist_bip_sci is None and arr.shape[0] > 1: dist_bip_sci = arr[[1], :]

        # collecter T possibles
        if dist_bip_cav is not None: Ts.append(dist_bip_cav.shape[1])
        if dist_bip_sci is not None: Ts.append(dist_bip_sci.shape[1])

        if len(Ts) == 0:
            continue
        T = int(min(Ts))

        # composer le bloc feature (12 + 2 = 14 canaux)  # <<< CHANGEMENT
        feat_blocks = []
        for inst in INST_ORDER:
            if mono[inst] is not None:
                feat_blocks.append(mono[inst][:, :T])  # (4,T)
            else:
                feat_blocks.append(np.zeros((4, T), dtype=float))  # <<< CHANGEMENT
        if dist_bip_cav is None: dist_bip_cav = np.zeros((1, T), dtype=float)
        else:                      dist_bip_cav = dist_bip_cav[:, :T]
        if dist_bip_sci is None: dist_bip_sci = np.zeros((1, T), dtype=float)
        else:                     dist_bip_sci = dist_bip_sci[:, :T]

        feats_CxT = np.vstack(feat_blocks + [dist_bip_cav, dist_bip_sci])  # (14, T)  # <<< CHANGEMENT
        feats_CxT_cln = clean_trial_CxT_no_norm(feats_CxT)

        # label (on prend le premier niveau valide rencontré dans les entrées de ce trial)
        lvl9 = None
        for inst in list(INST_ORDER) + ["bipolar_to_cavitron", "bipolar_to_scissors"]:
            it_list = inst_map.get(inst, [])
            lvl9 = _level_from_any(it_list)
            if lvl9 is not None:
                break
        if lvl9 is None or lvl9 not in LEVEL_TO_IDX:
            continue

        # infos
        co, pgy, rmask = to_coarse_and_pgy(lvl9)

        # fenêtrage
        for start in range(0, T-L+1, hop):
            seg_raw = feats_CxT[:, start:start+L]
            zero_ratio = float((seg_raw == 0).sum()) / float(seg_raw.size)
            if zero_ratio >= ZERO_RATIO_THRESHOLD:
                continue
            seg = feats_CxT_cln[:, start:start+L]
            X.append(seg.T.astype(np.float32))  # [L, C]
            y9.append(LEVEL_TO_IDX[lvl9])
            yco.append(co if (co is not None and 0<=co<4) else -1)
            ypgy.append(pgy if (0<=pgy<6) else -1)
            yresmask.append(rmask)
            parts.append(pid)
            trials.append(f"{pid}__{tr}")
            starts.append(start)
            inst_presence.append(presence.copy())
            inst_combo.append("|".join([INST_ORDER[i] for i,b in enumerate(presence) if b]))

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
    inst_presence = np.asarray(inst_presence, dtype=bool)  # shape: [N,3]
    inst_combo = np.asarray(inst_combo, object)

    # Trace des métriques conservées
    map_path = os.path.join(OUT_DIR, "metrics_used_full_multi.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("Index\tMetric name\n")
        for i, nm in enumerate(kept_names):
            f.write(f"{i}\t{nm}\n")

    print(f"Windows built = {len(X)} | L={L} | C={X.shape[2]} | classes=9")
    return X,y9,yco,ypgy,yresmask,parts,trials,starts,inst_presence,inst_combo, kept_names

def to_categorical_int(y, n):
    Y = np.zeros((len(y), n), dtype=np.float32)
    valid = (y >= 0) & (y < n)
    Y[np.where(valid)[0], y[valid]] = 1.0
    return Y

# ------------- Modèle -------------
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

# ------------- Métriques & Plots -------------
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
    xt = np.arange(len(col_labels)); yt = np.arange(len(row_labels))
    plt.xticks(xt, col_labels, rotation=45, ha="right")
    plt.yticks(yt, row_labels)
    plt.colorbar(); plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

# ---- Confusions par instrument (présence dans la fenêtre) ---
def save_confusions_by_instrument_presence(y_true_all: np.ndarray,
                                           y_pred_all: np.ndarray,
                                           inst_presence_all: np.ndarray,
                                           tag: str,
                                           out_dir: str = OUT_DIR):
    """
    inst_presence_all: [N,3] bool → colonnes = [bipolar, scissors, cavitron]
    On produit une matrice de confusion par instrument, en ne gardant que
    les fenêtres où l'instrument est présent (au moins un canal non-zero dans son bloc).
    """
    for j, inst in enumerate(INST_ORDER):
        idx = np.where(inst_presence_all[:, j])[0]
        if len(idx) == 0:
            print(f"[{tag}] Aucun window avec instrument={inst}")
            continue
        y_t = y_true_all[idx]
        y_p = y_pred_all[idx]
        cf  = confusion_matrix_np(y_t, y_p, n_classes=len(LEVEL9_ORDER))
        out = os.path.join(out_dir, f"confusion_windows_{tag}_inst-{inst}.png")
        plot_confusion(cf, LEVEL9_ORDER, outpath=out, title=f"Confusion — {tag} — instrument={inst}")
        print(f"[{tag}] confusion (présence) par instrument sauvée → {out}")

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

# ------------- Main (LOOCV by trials) -------------
def main():
    print(f"📁 Loading: {DATA_PATH}")
    items = load_items(DATA_PATH)
    (X_raw, y9, yco_idx, ypgy_idx, yresmask,
     part_ids, trial_ids, start_idx, inst_presence, inst_combo, kept_names) = build_windows_dataset_multi(
        items, L=SEQ_LEN, hop=HOP
    )
    N,L,C = X_raw.shape
    print(f"→ Final features (C={C}):")
    for i,nm in enumerate(kept_names):
        if i < 5 or i >= C-5:  # échantillon
            print(f"  {i:03d}: {nm}")
    print(f"... total {C} canaux")

    # Holders OOF
    oof_pred9 = np.zeros(N, np.int32)
    oof_prob9 = np.zeros((N,9), np.float32)
    fold_id   = np.zeros(N, np.int32)

    uniq_trials = np.array(sorted(np.unique(trial_ids)))
    print(f"LOOCV by trials: {len(uniq_trials)} unique trials")

    # Boucle Leave-One-Out
    for fold_idx, val_trial in enumerate(uniq_trials, start=1):
        va = np.where(trial_ids == val_trial)[0]
        tr = np.where(trial_ids != val_trial)[0]
        print(f"\n=== LOO fold {fold_idx}/{len(uniq_trials)} — val trial: {val_trial} ===")

        Xtr_raw, Xva_raw = X_raw[tr], X_raw[va]
        y9tr, y9va = y9[tr], y9[va]

        # Heads (coarse/fine) + masque résident
        def build_heads(idxs):
            Yco = np.zeros((len(idxs),4), np.float32)
            Ypg = np.zeros((len(idxs),6), np.float32)
            Msk = np.zeros((len(idxs),),  np.float32)
            for i, gi in enumerate(idxs):
                name = LEVEL9_ORDER[int(y9[gi])]
                co,pgy,rm = to_coarse_and_pgy(name)
                if co is not None and 0<=co<4: Yco[i,co]=1.0
                if 0<=pgy<6: Ypg[i,pgy]=1.0
                Msk[i]=rm
            return Yco, Ypg, Msk

        Yco_tr, Ypgy_tr, mask_tr = build_heads(tr)
        Yco_va, Ypgy_va, mask_va = build_heads(va)

        # Normalisation train-only
        mean_c, std_c = compute_train_norm_stats(Xtr_raw)
        Xtr = apply_norm(Xtr_raw, mean_c, std_c)
        Xva = apply_norm(Xva_raw, mean_c, std_c)

        # Équilibrage simple par sur-échantillonnage sur y9
        idx_by_c = {c: np.where(y9tr==c)[0] for c in range(9)}
        sizes = {c: len(v) for c,v in idx_by_c.items() if len(v)>0}
        max_n = max(sizes.values()) if sizes else 0
        rng_local = np.random.default_rng(SEED+fold_idx)
        new_tr_idx_rel = []
        for c, idxs_rel in idx_by_c.items():
            if len(idxs_rel)==0: continue
            need = max_n - len(idxs_rel)
            if need > 0:
                add = rng_local.choice(idxs_rel, size=need, replace=True)
                idxs_rel = np.concatenate([idxs_rel, add])
            new_tr_idx_rel.extend(list(idxs_rel))
        new_tr_idx_rel = np.array(new_tr_idx_rel, np.int64)
        rng_local.shuffle(new_tr_idx_rel)

        # Remap des têtes/masques
        Xtr = Xtr[new_tr_idx_rel]
        y9tr_bal = y9tr[new_tr_idx_rel]
        Yco_tr = Yco_tr[new_tr_idx_rel]
        Ypgy_tr= Ypgy_tr[new_tr_idx_rel]
        mask_tr= mask_tr[new_tr_idx_rel]

        # Modèle
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

        # Validation sur le trial laissé de côté
        p_coarse, p_fine = predict_heads(model, Xva)
        P9_va = reconstruct_level9_from_heads(p_coarse, p_fine)
        yhat9_va = np.argmax(P9_va, axis=1)

        oof_pred9[va] = yhat9_va
        oof_prob9[va] = P9_va
        fold_id[va]   = fold_idx

        acc = (y9va == yhat9_va).mean()
        bacc= balanced_accuracy_np(y9va, yhat9_va, 9)
        print(f"[LOO {fold_idx}] acc9={acc:.4f} | bal_acc9={bacc:.4f}")

    # ===== OOF (windows) =====
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

    # ➜ Matrices de confusion par instrument (présence dans la fenêtre)
    save_confusions_by_instrument_presence(
        y_true_all=y9, y_pred_all=oof_pred9, inst_presence_all=inst_presence, tag="oof"
    )

    # CSV (windows OOF avec proba)
    win_csv = os.path.join(OUT_DIR,"pred_windows_oof.csv")
    with open(win_csv,"w",encoding="utf-8") as f:
        header=["fold","participant","trial","start","inst_combo","true9","pred9"]+[f"p_{c}" for c in LEVEL9_ORDER]
        f.write(",".join(header)+"\n")
        for i in range(len(y9)):
            row=[str(int(fold_id[i])), str(part_ids[i]), str(trial_ids[i]), str(int(start_idx[i])),
                 str(inst_combo[i]),
                 LEVEL9_ORDER[int(y9[i])], LEVEL9_ORDER[int(oof_pred9[i])]] + [f"{float(oof_prob9[i,j]):.6f}" for j in range(9)]
            f.write(",".join(row)+"\n")

    # Version agrégée par participant-trial
    print("\n📊 Generating grouped OOF predictions by participant-trial...")
    group_predictions_by_trial(win_csv)

    # Participant-level (majorité) à partir de l’OOF
    d_pred_part = majority_aggregate(oof_pred9, part_ids)
    d_true_part = mode_true_by_entity(y9, part_ids)
    ents, ytrue_part, ypred_part = dict_to_arrays(d_pred_part, d_true_part)
    rep_part = classification_report_simple(ytrue_part, ypred_part, LEVEL9_ORDER)
    with open(os.path.join(OUT_DIR,"report_participants_oof.txt"),"w",encoding="utf-8") as f: f.write(rep_part)
    print("\n=== OOF participants (majority, 9 classes) ===\n"+rep_part)
    cf_part = confusion_matrix_np(ytrue_part, ypred_part, 9)
    plot_confusion(cf_part, LEVEL9_ORDER, os.path.join(OUT_DIR,"confusion_participants_majority_oof.png"),
                   title="Confusion — participants (majority, OOF)")

    print("\n✅ Done. Check outputs under:", OUT_DIR)

if __name__ == "__main__":
    main()
