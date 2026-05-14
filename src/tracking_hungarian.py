"""
tracking_hungarian.py
=====================
Pipeline de tracking propre par algorithme Hongrois.
Remplace le centroïde naïf par suivi persistant de chaque sphère.

Usage :
    python pipeline/tracking_hungarian.py --seq dataset/2026-04-24/bipolar/move/R09
    python pipeline/tracking_hungarian.py --all dataset/2026-04-24  --report
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # pas besoin d'écran
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# ─── CONSTANTES ────────────────────────────────────────────────────────────────
N_SPHERES   = {"bipolar": 1, "scissors": 1, "aspirator": 1}
MAX_JUMP_MM = 50.0    # saut max entre 2 frames (physiquement plausible @ 14Hz)
TRI_ERR_MAX = 15.0    # triangulation_error max acceptable
Z_ABS_MAX   = 2000.0  # |z| > 2m = outlier évident
MIN_FRAMES  = 30      # séquence trop courte → rejetée

# ─── ÉTAPE A : Nettoyage outliers physiques ────────────────────────────────────
def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les détections physiquement impossibles AVANT le tracking.
    On garde uniquement les points avec :
      - triangulation_error < TRI_ERR_MAX (5 mm)
      - probability >= 0.9 (détection fiable)
      - |z| < 2000 mm (pas un fantôme optique)
      - |x|, |y| < 2000 mm
    """
    mask = (
        (df["triangulation_error"] < TRI_ERR_MAX) &
        (df["probability"] >= 0.9) &
        (df["z"].abs() < Z_ABS_MAX) &
        (df["x"].abs() < Z_ABS_MAX) &
        (df["y"].abs() < Z_ABS_MAX)
    )
    cleaned = df[mask].copy().reset_index(drop=True)
    return cleaned


# ─── ÉTAPE B : Algorithme Hongrois ────────────────────────────────────────────
def hungarian_assign(prev_xyz: np.ndarray, curr_xyz: np.ndarray,
                     max_jump: float = MAX_JUMP_MM):
    """
    Associe chaque piste précédente à la détection la plus proche (coût = distance).
    
    Mathématiques :
        Matrice de coût C[i,j] = ||prev[i] - curr[j]||₂
        Problème d'assignement : min Σ C[i,j]  s.c. bijection
        Résolu par scipy.optimize.linear_sum_assignment (algorithme Hongrois, O(n³))
    
    prev_xyz : [n_tracks, 3]  positions précédentes des pistes
    curr_xyz : [n_det, 3]     positions des nouvelles détections
    max_jump : seuil de distance (mm) au-delà duquel l'association est rejetée
    
    Retourne : dict {track_idx → detection_idx}  (ou absent si piste perdue)
    """
    if len(prev_xyz) == 0 or len(curr_xyz) == 0:
        return {}

    # Matrice de coût [n_tracks × n_det]
    diff   = prev_xyz[:, None, :] - curr_xyz[None, :, :]   # broadcasting
    cost   = np.linalg.norm(diff, axis=2)                   # distances euclidiennes

    row_ind, col_ind = linear_sum_assignment(cost)

    assignment = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_jump:          # n'associer que si physiquement plausible
            assignment[int(r)] = int(c)

    return assignment


# ─── ÉTAPE C : Tracking complet d'une séquence ────────────────────────────────
def track_sequence(csv_clean_path: Path, n_spheres: int):
    """
    Applique le tracking Hongrois à une séquence complète.
    
    Retourne :
        df_tracks : DataFrame [frame_idx, sphere_id, x, y, z, valid]
        stats     : dict de métriques de qualité
    """
    df = pd.read_csv(csv_clean_path)
    df = clean_outliers(df)
    frames_sorted = sorted(df["frame_idx"].unique())

    if len(frames_sorted) < MIN_FRAMES:
        return None, {"error": f"Trop peu de frames : {len(frames_sorted)}"}

    # ── Initialisation des pistes ──────────────────────────────────────────────
    # On cherche la première frame avec >= n_spheres détections valides
    tracks = {}    # {sphere_id: np.array([x,y,z])}
    start_frame = None

    for fi in frames_sorted:
        frame_det = df[df["frame_idx"] == fi]
        if len(frame_det) >= n_spheres:
            # Prendre les n meilleures détections (par triangulation_error croissante)
            best = frame_det.nsmallest(n_spheres, "triangulation_error")
            for sid in range(n_spheres):
                row = best.iloc[sid]
                tracks[sid] = np.array([row["x"], row["y"], row["z"]])
            start_frame = fi
            break

    if start_frame is None:
        return None, {"error": "Aucune frame avec assez de sphères valides"}

    # ── Tracking frame par frame ───────────────────────────────────────────────
    records = []
    jump_distances = []    # pour les métriques
    lost_frames   = 0

    for fi in frames_sorted:
        frame_det = df[df["frame_idx"] == fi]

        if fi == start_frame:
            # Déjà initialisé
            for sid, xyz in tracks.items():
                records.append({
                    "frame_idx": fi, "sphere_id": sid,
                    "x": xyz[0], "y": xyz[1], "z": xyz[2], "valid": 1
                })
            continue

        # Préparer les arrays pour l'assignement
        track_ids  = sorted(tracks.keys())
        prev_xyz   = np.array([tracks[sid] for sid in track_ids])
        curr_xyz   = frame_det[["x", "y", "z"]].values if len(frame_det) > 0 else np.empty((0, 3))

        assignment = hungarian_assign(prev_xyz, curr_xyz)

        for i, sid in enumerate(track_ids):
            if i in assignment:
                det_row = frame_det.iloc[assignment[i]]
                new_xyz = np.array([det_row["x"], det_row["y"], det_row["z"]])

                jump = float(np.linalg.norm(new_xyz - tracks[sid]))
                jump_distances.append(jump)

                tracks[sid] = new_xyz
                records.append({
                    "frame_idx": fi, "sphere_id": sid,
                    "x": new_xyz[0], "y": new_xyz[1], "z": new_xyz[2], "valid": 1
                })
            else:
                # Piste perdue : on garde la dernière position connue
                xyz = tracks[sid]
                records.append({
                    "frame_idx": fi, "sphere_id": sid,
                    "x": xyz[0], "y": xyz[1], "z": xyz[2], "valid": 0
                })
                lost_frames += 1

    df_tracks = pd.DataFrame(records)

    n_total = len(frames_sorted) * n_spheres
    stats = {
        "n_frames":       len(frames_sorted),
        "n_spheres":      n_spheres,
        "lost_events":    lost_frames,
        "pct_valid":      round(100 * (n_total - lost_frames) / n_total, 1),
        "jump_mean_mm":   round(float(np.mean(jump_distances)), 2) if jump_distances else None,
        "jump_max_mm":    round(float(np.max(jump_distances)), 2)  if jump_distances else None,
        "jump_p95_mm":    round(float(np.percentile(jump_distances, 95)), 2) if jump_distances else None,
    }
    return df_tracks, stats


# ─── ÉTAPE D : Calcul des 6 features cinématiques ─────────────────────────────
def compute_features(df_tracks: pd.DataFrame, dt: float = 1/14.14):
    """
    Calcule les 6 features SANS pos_mag (invariants à la pose caméra).
    
    Features :
        velocity     = ||Δcentroïde|| / dt           (mm/s)
        acceleration = ||Δvelocity|| / dt            (mm/s²)
        jerk         = ||Δacceleration|| / dt        (mm/s³)
        spread       = mean(||si - sj|| pour i≠j)    (mm) — rigidité de l'outil
        axis_angle   = angle(premier axe PCA)        (radians)
        valid_ratio  = sphères valides / total        (sans unité)
    
    Retourne : array [T, 6]
    """
    frames = sorted(df_tracks["frame_idx"].unique())
    T = len(frames)

    # Pivot : positions par frame [T, n_spheres, 3]
    n_sph = df_tracks["sphere_id"].nunique()
    pos_arr   = np.zeros((T, n_sph, 3))
    valid_arr = np.zeros((T, n_sph))

    for t, fi in enumerate(frames):
        f = df_tracks[df_tracks["frame_idx"] == fi].sort_values("sphere_id")
        for s in range(n_sph):
            if s < len(f):
                row = f.iloc[s]
                pos_arr[t, s]   = [row["x"], row["y"], row["z"]]
                valid_arr[t, s] = row["valid"]

    # ── Feature 1–3 : cinématique du centroïde ──────────────────────────────
    # Centroïde pondéré par valid (ignore les sphères perdues)
    w = valid_arr / (valid_arr.sum(axis=1, keepdims=True) + 1e-9)  # [T, n_sph]
    centroid = (pos_arr * w[:, :, None]).sum(axis=1)                # [T, 3]

    delta    = np.diff(centroid, axis=0, prepend=centroid[[0]])
    velocity = np.linalg.norm(delta, axis=1) / dt                   # [T]

    d_vel    = np.diff(velocity, prepend=velocity[0])
    accel    = np.abs(d_vel) / dt                                    # [T]

    d_acc    = np.diff(accel, prepend=accel[0])
    jerk     = np.abs(d_acc) / dt                                    # [T]

    # ── Feature 4 : spread (écartement des sphères) ────────────────────────
    # Mesure la rigidité : un outil rigide a un spread constant.
    # Variation du spread = flexion anormale de l'outil ou glissement.
    spread = np.zeros(T)
    for t in range(T):
        valid_pos = pos_arr[t][valid_arr[t] > 0]
        if len(valid_pos) >= 2:
            dists = []
            for i in range(len(valid_pos)):
                for j in range(i+1, len(valid_pos)):
                    dists.append(np.linalg.norm(valid_pos[i] - valid_pos[j]))
            spread[t] = np.mean(dists)
        else:
            spread[t] = spread[t-1] if t > 0 else 0

    # ── Feature 5 : axis_angle (orientation de l'outil) ─────────────────────
    # Premier axe PCA des positions des sphères = axe principal de l'outil.
    # L'angle de cet axe avec l'axe Z encode l'inclinaison de l'outil.
    axis_angle = np.zeros(T)
    for t in range(T):
        valid_pos = pos_arr[t][valid_arr[t] > 0]
        if len(valid_pos) >= 2:
            centered = valid_pos - valid_pos.mean(axis=0)
            if centered.std() > 1e-9:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                principal = Vt[0]
                angle = np.arccos(np.clip(
                    np.abs(np.dot(principal, [0, 0, 1])), 0, 1
                ))
                axis_angle[t] = angle
        else:
            axis_angle[t] = axis_angle[t-1] if t > 0 else 0

    # ── Feature 6 : valid_ratio ─────────────────────────────────────────────
    valid_ratio = valid_arr.mean(axis=1)    # [T]

    features = np.stack(
        [velocity, accel, jerk, spread, axis_angle, valid_ratio],
        axis=1
    ).astype(np.float32)    # [T, 6]

    return features, frames


# ─── ÉTAPE E : Visualisation et rapport ───────────────────────────────────────
def plot_comparison(df_tracks, features, seq_path, out_dir):
    """
    Génère 3 figures de diagnostic :
    1. Trajectoires 3D des sphères (avant = centroïde naïf, après = pistes Hongrois)
    2. Jump distances frame-à-frame (doit être < 25 mm)
    3. Les 6 features sur la durée de la séquence
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_name = Path(seq_path).name

    # ── Figure 1 : trajectoires 3D des sphères ──────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"Tracking Hongrois — {seq_name}", fontsize=12)

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Avant : centroïde naïf (mélange de sphères)")
    # Simuler l'ancien pipeline (centroïde de tout)
    raw = pd.read_csv(Path(seq_path) / "fiducials_clean.csv")
    naive_x = raw.groupby("frame_idx")["x"].mean().values
    naive_y = raw.groupby("frame_idx")["y"].mean().values
    naive_z = raw.groupby("frame_idx")["z"].mean().values
    ax1.plot(naive_x, naive_y, naive_z, "r-", lw=0.8, alpha=0.6)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Après : pistes Hongrois (sphères séparées)")
    colors = ["#2166ac", "#1a9850", "#d73027", "#f4a582", "#8e44ad"]
    for sid in sorted(df_tracks["sphere_id"].unique()):
        t = df_tracks[(df_tracks["sphere_id"] == sid) & (df_tracks["valid"] == 1)]
        ax2.plot(t["x"].values, t["y"].values, t["z"].values,
                 color=colors[sid % len(colors)], lw=1.0, label=f"Sphère {sid}")
    ax2.legend(fontsize=7); ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(out_dir / f"{seq_name}_3d_comparison.png", dpi=150)
    plt.close()

    # ── Figure 2 : jump distances ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Déplacements frame-à-frame — {seq_name}", fontsize=11)

    # Naïf
    naive_pos = np.stack([naive_x, naive_y, naive_z], axis=1)
    naive_jumps = np.linalg.norm(np.diff(naive_pos, axis=0), axis=1)
    axes[0].plot(naive_jumps, color="#d73027", lw=0.8)
    axes[0].axhline(25, color="gray", ls="--", lw=1, label="Seuil 25mm")
    axes[0].set_title(f"Centroïde naïf — moy={naive_jumps.mean():.1f}mm")
    axes[0].set_ylabel("mm"); axes[0].legend()

    # Hongrois (centroïde des pistes valides)
    frames = sorted(df_tracks["frame_idx"].unique())
    n_sph = df_tracks["sphere_id"].nunique()
    hon_centroid = np.zeros((len(frames), 3))
    for t, fi in enumerate(frames):
        f = df_tracks[(df_tracks["frame_idx"] == fi) & (df_tracks["valid"] == 1)]
        if len(f) > 0:
            hon_centroid[t] = f[["x","y","z"]].mean().values
    hon_jumps = np.linalg.norm(np.diff(hon_centroid, axis=0), axis=1)
    axes[1].plot(hon_jumps, color="#1a9850", lw=0.8)
    axes[1].axhline(25, color="gray", ls="--", lw=1, label="Seuil 25mm")
    axes[1].set_title(f"Hongrois — moy={hon_jumps.mean():.1f}mm")
    axes[1].set_ylabel("mm"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"{seq_name}_jumps.png", dpi=150)
    plt.close()

    # ── Figure 3 : les 6 features ────────────────────────────────────────────
    feat_names = ["velocity (mm/s)", "accel (mm/s²)", "jerk (mm/s³)",
                  "spread (mm)", "axis_angle (rad)", "valid_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"6 features cinématiques — {seq_name}", fontsize=11)
    for i, (name, ax) in enumerate(zip(feat_names, axes.flatten())):
        ax.plot(features[:, i], color="#2166ac", lw=0.9)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / f"{seq_name}_features.png", dpi=150)
    plt.close()

    print(f"  → Figures sauvegardées dans {out_dir}")


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────
def process_one(seq_path: str, out_dir: str = "pipeline_output"):
    seq_path = Path(seq_path)
    instrument = seq_path.parts[-3]    # ex: "bipolar"
    n_sph = N_SPHERES.get(instrument, 4)

    print(f"\n{'='*60}")
    print(f"Séquence : {seq_path}")
    print(f"Instrument : {instrument}  ({n_sph} sphères attendues)")

    csv_path = seq_path / "fiducials_clean.csv"
    if not csv_path.exists():
        print(f"  ✗ fiducials_clean.csv introuvable")
        return None, None

    df_tracks, stats = track_sequence(csv_path, n_sph)

    if df_tracks is None:
        print(f"  ✗ Tracking échoué : {stats.get('error')}")
        return None, stats

    print(f"  Frames trackées    : {stats['n_frames']}")
    print(f"  % pistes valides   : {stats['pct_valid']}%")
    print(f"  Jump moyen         : {stats['jump_mean_mm']} mm")
    print(f"  Jump max           : {stats['jump_max_mm']} mm")
    print(f"  Jump P95           : {stats['jump_p95_mm']} mm")

    features, _ = compute_features(df_tracks)
    print(f"  Features shape     : {features.shape}  [T × 6]")

    # Sauvegarder les résultats
    out = Path(out_dir) / instrument / seq_path.parts[-2] / seq_path.parts[-1]
    out.mkdir(parents=True, exist_ok=True)
    df_tracks.to_csv(out / "tracks_hungarian.csv", index=False)
    np.save(out / "features_6ch.npy", features)

    # Plots
    plot_comparison(df_tracks, features, seq_path, out)

    # Rapport JSON
    stats["features_shape"] = list(features.shape)
    stats["seq_path"] = str(seq_path)
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return df_tracks, stats


def process_all(dataset_path: str, out_dir: str = "pipeline_output"):
    """Lance le tracking sur toutes les séquences du dataset."""
    base = Path(dataset_path)
    all_stats = []

    for instrument in ["bipolar", "scissors", "aspirator"]:
        for motion in ["static", "move", "rotate", "cycle"]:
            seq_dir = base / instrument / motion
            if not seq_dir.exists():
                continue
            for rep in sorted(seq_dir.iterdir()):
                if rep.is_dir() and rep.name.startswith("R"):
                    _, stats = process_one(str(rep), out_dir)
                    if stats and "error" not in stats:
                        stats["instrument"] = instrument
                        stats["motion"]     = motion
                        stats["rep"]        = rep.name
                        all_stats.append(stats)

    # Rapport global
    if all_stats:
        df_report = pd.DataFrame(all_stats)
        df_report.to_csv(Path(out_dir) / "global_report.csv", index=False)
        print(f"\n{'='*60}")
        print(f"RAPPORT GLOBAL — {len(all_stats)} séquences traitées")
        print(df_report[["instrument","motion","rep",
                          "pct_valid","jump_mean_mm","jump_p95_mm"]].to_string())
        print(f"\nJump moyen global : {df_report['jump_mean_mm'].mean():.2f} mm")
        print(f"% valide moyen    : {df_report['pct_valid'].mean():.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",    help="Chemin d'une séquence unique")
    parser.add_argument("--all",    help="Chemin du dataset complet")
    parser.add_argument("--out",    default="pipeline_output")
    args = parser.parse_args()

    if args.seq:
        process_one(args.seq, args.out)
    elif args.all:
        process_all(args.all, args.out)
    else:
        print("Usage : --seq <chemin> ou --all <dataset>")
