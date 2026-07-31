"""
diagnose_tsne_baseline.py
=========================
Étape 0 — Diagnostic t-SNE des embeddings baseline (OBLIGATOIRE avant entraînement).

Mesure la séparabilité Senior vs Expert dans l'espace des features cinématiques :
  Distance centroïdes < 5  → signal structurel (publier baseline)
  Distance centroïdes ≥ 5  → signal exploitable (poursuivre approches 1–4)

Usage (depuis ICEMS-main) :
    python src/diagnose_tsne_baseline.py
    python src/diagnose_tsne_baseline.py --data data/continuous_per_trial.pkl --out results/diagnostic_tsne
    python src/diagnose_tsne_baseline.py --checkpoint results/lopo_corrected/.../  # embeddings modèle si dispo
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from icems_strategy import (  # noqa: E402
    CENTROID_DISTANCE_THRESHOLD,
    CLASS4_NAMES,
    interpret_centroid_distance,
    senior_expert_centroid_distance,
)
from step_B_classification import (  # noqa: E402
    Y9_TO_Y4,
    _normalize_dataset,
    apply_norm,
    compute_norm_stats,
    enrich_kinematic_features,
    frame_valid_mask,
)

VALID_COL = 9


def load_dataset(path: Path) -> Dict:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return _normalize_dataset(raw)


def extract_raw_embeddings(dataset: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Embeddings baseline = mean+std cinématique par trial (sans entraînement)."""
    keys = sorted(dataset.keys())
    embs, y4, y9, pids = [], [], [], []
    for key in keys:
        rec = dataset[key]
        X = enrich_kinematic_features(np.asarray(rec["X"], dtype=np.float32))
        valid = X[:, VALID_COL] > 0
        if not valid.any():
            valid = np.ones(X.shape[0], dtype=bool)
        kin = X[valid, :VALID_COL]
        emb = np.concatenate([kin.mean(axis=0), kin.std(axis=0)])
        embs.append(emb)
        y9_i = int(rec["y9"])
        y4_i = int(rec.get("y4", Y9_TO_Y4[y9_i]))
        y4.append(y4_i)
        y9.append(y9_i)
        pids.append(str(key[0]))
    return (
        np.stack(embs).astype(np.float64),
        np.asarray(y4, dtype=int),
        np.asarray(y9, dtype=int),
        pids,
    )


def extract_model_embeddings(dataset: Dict, device) -> Optional[np.ndarray]:
    """Embeddings trial_embedding d'un CausalGRU entraîné sur tout le dataset (diagnostic only)."""
    try:
        import torch
        from step_B_classification import CausalGRUScorer
    except ImportError:
        return None

    mean, std = compute_norm_stats(dataset)
    model = CausalGRUScorer().to(device)
    model.eval()
    embs = []
    with torch.no_grad():
        for key in sorted(dataset.keys()):
            rec = dataset[key]
            x = apply_norm(rec["X"], mean, std)
            xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
            vm = torch.from_numpy(frame_valid_mask(x)).unsqueeze(0).to(device)
            emb = model.trial_embedding(xt, vm).cpu().numpy()[0]
            embs.append(emb)
    return np.stack(embs).astype(np.float64)


def run_tsne(embeddings: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    n = len(embeddings)
    perp = min(perplexity, max(2, (n - 1) // 3))
    emb_std = StandardScaler().fit_transform(embeddings)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(emb_std)


def plot_tsne(
    coords: np.ndarray,
    y4: np.ndarray,
    y9: np.ndarray,
    dist: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from step_B_classification import SUBLEVEL_SHORT

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    for c in range(4):
        mask = y4 == c
        if not mask.any():
            continue
        axes[0].scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[c], label=f"{CLASS4_NAMES[c]} (n={mask.sum()})",
            alpha=0.75, s=55, edgecolors="black", linewidths=0.3,
        )
    axes[0].set_title(f"t-SNE — 4 classes\nDistance Senior/Expert = {dist:.2f}")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)

    cmap = plt.get_cmap("RdYlGn")
    for y9_i in range(9):
        mask = y9 == y9_i
        if not mask.any():
            continue
        axes[1].scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(y9_i / 8.0)], label=f"{SUBLEVEL_SHORT[y9_i]} (n={mask.sum()})",
            alpha=0.75, s=55, edgecolors="black", linewidths=0.3,
        )
    axes[1].set_title("t-SNE — 9 sous-niveaux")
    axes[1].legend(loc="best", fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    decision = interpret_centroid_distance(dist)
    fig.suptitle(
        f"Diagnostic baseline — seuil={CENTROID_DISTANCE_THRESHOLD} → {decision}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def print_report(
    embeddings: np.ndarray,
    y4: np.ndarray,
    dist: float,
    mode: str,
) -> dict:
    print("\n" + "=" * 60)
    print(" ÉTAPE 0 — Diagnostic t-SNE baseline")
    print("=" * 60)
    print(f"\n[Mode]     {mode}")
    print(f"[Trials]   n={len(embeddings)}  dim={embeddings.shape[1]}")

    print("\n[Distribution y4]")
    for c, name in enumerate(CLASS4_NAMES):
        print(f"  {name:<8} n={(y4 == c).sum()}")

    senior_mask = y4 == 2
    expert_mask = y4 == 3
    print(f"\n[Séparabilité Senior vs Expert]")
    print(f"  Distance centroïdes (L2, features standardisées) : {dist:.3f}")
    print(f"  Seuil décision                                : {CENTROID_DISTANCE_THRESHOLD}")
    decision = interpret_centroid_distance(dist)
    print(f"  → DÉCISION : {decision}")

    if dist < CENTROID_DISTANCE_THRESHOLD:
        print("\n  ⚠ Signal structurel — les approches 2–4 risquent de répéter V2b.")
        print("    Recommandation : publier la baseline (r_participant=0.870).")
    else:
        print("\n  ✓ Signal exploitable — dérouler les approches dans l'ordre (1→4).")
        print("    Commencer par : python run_lopo_corrected.py --approach 1 --smoke-test")

    return {
        "mode": mode,
        "n_trials": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "senior_expert_centroid_distance": dist,
        "threshold": CENTROID_DISTANCE_THRESHOLD,
        "decision": decision,
        "continue_experiments": dist >= CENTROID_DISTANCE_THRESHOLD,
        "class_counts": {CLASS4_NAMES[c]: int((y4 == c).sum()) for c in range(4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostic t-SNE baseline Senior/Expert.")
    ap.add_argument("--data", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/diagnostic_tsne"))
    ap.add_argument("--perplexity", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--model-embeddings",
        action="store_true",
        help="Ajoute des embeddings CausalGRU non entraîné (initialisation aléatoire).",
    )
    args = ap.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} introuvable.\n"
            "Lancez d'abord : python src/build_continuous_dataset.py"
        )

    dataset = load_dataset(args.data)
    embs_raw, y4, y9, _ = extract_raw_embeddings(dataset)
    dist, _, _ = senior_expert_centroid_distance(embs_raw, y4)

    report = print_report(embs_raw, y4, dist, mode="kinematic_mean_std")

    coords = run_tsne(embs_raw, args.perplexity, args.seed)
    plot_path = args.out / "tsne_baseline_4class.png"
    plot_tsne(coords, y4, y9, dist, plot_path)
    print(f"\n[Figure] {plot_path.resolve()}")

    if args.model_embeddings:
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embs_model = extract_model_embeddings(dataset, device)
            if embs_model is not None:
                dist_m, _, _ = senior_expert_centroid_distance(embs_model, y4)
                print(f"\n[Modèle non entraîné] distance Senior/Expert = {dist_m:.3f}")
                report["model_untrained_distance"] = dist_m
        except Exception as exc:
            print(f"\n[WARN] Embeddings modèle ignorés : {exc}")

    report_path = args.out / "tsne_diagnostic.json"
    args.out.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[JSON]   {report_path.resolve()}")


if __name__ == "__main__":
    main()
