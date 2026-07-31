"""
ablation_atracsys.py
====================
Ablation MAE Atracsys vs init aléatoire — pipeline ICEMS (Solution C).

Condition A (with_atracsys) : encodeur MAE pré-entraîné (best_encoder.pt, Run 8).
Condition B (without_atracsys) : même architecture, poids PyTorch par défaut.

Les deux conditions partagent le même protocole LOPO 47-fold, mêmes hyperparamètres
et la même seed. Seule l'initialisation de l'encodeur diffère.

Usage (depuis ICEMS-main) :
    python src/ablation_atracsys.py --data data/continuous_per_trial.pkl --out results/ablation_atracsys
    python src/ablation_atracsys.py --encoder ~/icems/results/mae_run8/best_encoder.pt --max-folds 3

Sortie :
    results/ablation_atracsys/ablation_results.json
    results/ablation_atracsys/A/  et  B/  (prédictions LOPO par condition)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, wilcoxon

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# Import du pipeline LOPO existant (pas de duplication).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_B_classification import (  # noqa: E402
    save_lopo_results,
    load_lopo_results,
    print_metrics,
    print_early_stopping_summary,
    plot_score_vs_time,
    plot_confusion,
    plot_scatter,
    LOPO_PREDICTIONS_FILE,
    trial_y4_reg,
)
from mae_encoder import (  # noqa: E402
    MAEEncoder4ch,
    MultiInstrumentMAEFront,
    load_encoder_checkpoint,
)
import step_B_classification as step_b  # noqa: E402


SEED = 42
PRACTICAL_DELTA_R = 0.05


class CausalGRUScorerMAE(step_b.CausalGRUScorer):
    """
    GRU causal avec front MAE Solution C (3 instruments × 64 dims).
    Même tête GRU que CausalGRUScorer ; seul le front diffère.
    """

    def __init__(
        self,
        encoder: Optional[MAEEncoder4ch] = None,
        encoder_ckpt: Optional[Path] = None,
        freeze_encoder: bool = False,
        dropout: float = step_b.TRAIN_DROPOUT,
    ):
        nn.Module.__init__(self)
        if encoder is None:
            encoder = MAEEncoder4ch()
        self.mae_front = MultiInstrumentMAEFront(encoder)
        front_dim = self.mae_front.out_dim

        if encoder_ckpt is not None:
            missing, unexpected = load_encoder_checkpoint(
                self.mae_front.encoder, encoder_ckpt, freeze=freeze_encoder,
            )
            print(f"  [MAE] chargé depuis {encoder_ckpt}")
            if missing:
                print(f"        clés manquantes ({len(missing)}) : {missing[:5]}...")
            if unexpected:
                print(f"        clés inattendues ({len(unexpected)}) : {unexpected[:5]}...")

        self.proj = torch.nn.Linear(front_dim + 1, 64)
        self.gru = torch.nn.GRU(64, 64, num_layers=2, dropout=dropout, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        z_mae = self.mae_front(x, valid_mask)
        inp = torch.cat([z_mae, valid_mask.unsqueeze(-1)], dim=-1)
        z = torch.nn.functional.gelu(self.proj(inp))
        out, _ = self.gru(z)
        return self.head(out).squeeze(-1)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_fold_with_encoder(
    train_trials: Dict,
    val_trials: Dict,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    batch_size: int,
    encoder_ckpt: Optional[Path],
    freeze_encoder: bool,
) -> Tuple[CausalGRUScorerMAE, dict]:
    """Réutilise train_fold mais instancie CausalGRUScorerMAE."""
    mean, std = step_b.compute_norm_stats(train_trials)
    train_items = step_b.trials_to_items(train_trials, mean, std)
    val_items = step_b.trials_to_items(val_trials, mean, std)

    val_loader = torch.utils.data.DataLoader(
        step_b.TrialDataset(val_items),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=step_b.collate_trials,
    )

    model = CausalGRUScorerMAE(
        encoder_ckpt=encoder_ckpt,
        freeze_encoder=freeze_encoder,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, wait = float("inf"), None, 0
    best_epoch, stopped_epoch = 0, epochs

    for ep in range(1, epochs + 1):
        model.train()
        step_b.set_dropout_rate(model, step_b.TRAIN_DROPOUT)
        stratified_items = step_b.build_stratified_items(train_items)
        loader = torch.utils.data.DataLoader(
            step_b.TrialDataset(stratified_items),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=step_b.collate_trials,
        )
        for xs, masks, y, y4_classes in loader:
            xs, masks, y = xs.to(device), masks.to(device), y.to(device)
            y4_classes = y4_classes.to(device)
            scores = model(xs, masks)
            agg = step_b.aggregate_score(scores, masks)
            loss = step_b.ordinal_loss(agg, y, y4_classes)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xs, masks, y, y4_val in val_loader:
                xs, masks, y = xs.to(device), masks.to(device), y.to(device)
                y4_val = y4_val.to(device)
                scores = model(xs, masks)
                agg = step_b.aggregate_score(scores, masks)
                val_losses.append(
                    step_b.ordinal_loss(agg, y, y4_val).item()
                )
        vloss = float(np.mean(val_losses)) if val_losses else float("inf")

        if vloss < best_val - 1e-5:
            best_val = vloss
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = ep
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model._norm_mean = mean
    model._norm_std = std
    train_info = {
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "early_stopped": stopped_epoch < epochs,
        "best_val_loss": best_val,
    }
    return model, train_info


def run_lopo_mae(
    dataset: Dict,
    device: torch.device,
    epochs: int,
    max_folds: Optional[int],
    patience: int,
    batch_size: int,
    mc_passes: int,
    aug_target: Optional[int],
    no_inline_aug: bool,
    encoder_ckpt: Optional[Path],
    freeze_encoder: bool,
    seed: int,
) -> Tuple[List[dict], Dict]:
    """LOPO identique à step_B, avec encodeur MAE optionnel."""
    dataset = step_b._normalize_dataset(dataset)
    by_pid = defaultdict(list)
    for k in dataset:
        by_pid[k[0]].append(k)

    pids = sorted(by_pid.keys())
    if max_folds is not None:
        pids = pids[:max_folds]

    all_preds = []
    curves = []
    fold_train_info: List[dict] = []

    for fold_i, held_pid in enumerate(pids):
        _set_seeds(seed + fold_i)
        print(f"\n[LOPO fold {fold_i + 1}/{len(pids)}] participant tenu out : {held_pid}")
        val_keys = set(by_pid[held_pid])
        train_real = {k: dataset[k] for k in dataset if k not in val_keys}
        if no_inline_aug:
            synth = {}
        else:
            synth = step_b._augment_train_fold_inline(
                train_real, fold_i=fold_i, aug_target=aug_target, seed=seed,
            )
        train = {**train_real, **synth}
        val = {k: dataset[k] for k in val_keys}

        model, train_info = _train_fold_with_encoder(
            train, val, device,
            epochs=epochs, lr=1e-3, weight_decay=1e-4,
            patience=patience, batch_size=batch_size,
            encoder_ckpt=encoder_ckpt, freeze_encoder=freeze_encoder,
        )
        train_info["participant"] = held_pid
        train_info["fold"] = fold_i + 1
        fold_train_info.append(train_info)

        if train_info["early_stopped"]:
            print(
                f"  Early stopping epoch {train_info['stopped_epoch']}/{epochs} "
                f"(best={train_info['best_epoch']}, val_loss={train_info['best_val_loss']:.5f})"
            )

        mean, std = model._norm_mean, model._norm_std
        for key in val_keys:
            rec = dataset[key]
            x = step_b.apply_norm(rec["X"], mean, std)
            mean_t, std_t, score, unc = step_b.predict_trial_mc(
                model, x, device, n_passes=mc_passes,
            )
            y4, y_reg = trial_y4_reg(rec)
            pred_cls = step_b.score_to_class(score)
            all_preds.append({
                "key": key,
                "y_reg": y_reg,
                "y4": y4,
                "score": score,
                "uncertainty": unc,
                "pred_class": pred_cls,
                "participant": held_pid,
                "fold": fold_i + 1,
            })
            curves.append({
                "t_norm": np.linspace(0, 1, len(mean_t)),
                "mean": mean_t,
                "std": std_t,
                "y4": y4,
                "key": key,
            })

    return all_preds, {"curves": curves, "fold_train_info": fold_train_info}


def global_metrics(preds: List[dict]) -> Dict[str, float]:
    y_true = np.array([p["y_reg"] for p in preds], dtype=float)
    y_pred = np.array([p["score"] for p in preds], dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "mae": mae,
        "n_trials": len(preds),
    }


def per_fold_metrics(preds: List[dict]) -> List[dict]:
    """Métriques par fold LOPO (participant tenu out)."""
    by_fold: Dict[int, List[dict]] = defaultdict(list)
    for p in preds:
        by_fold[int(p["fold"])].append(p)

    rows = []
    for fold in sorted(by_fold):
        items = by_fold[fold]
        y_true = np.array([x["y_reg"] for x in items], dtype=float)
        y_pred = np.array([x["score"] for x in items], dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        unc = float(np.mean([x["uncertainty"] for x in items]))

        if len(y_true) >= 2 and y_true.std() > 1e-9 and y_pred.std() > 1e-9:
            pr, _ = pearsonr(y_true, y_pred)
            sr, _ = spearmanr(y_true, y_pred)
        else:
            pr, sr = float("nan"), float("nan")

        rows.append({
            "fold": fold,
            "participant": items[0]["participant"],
            "n_trials": len(items),
            "r_pearson": float(pr),
            "r_spearman": float(sr),
            "mae": mae,
            "mean_uncertainty": unc,
        })
    return rows


def per_participant_metrics(preds: List[dict]) -> List[dict]:
    """Une ligne par participant (médiane des trials) — Wilcoxon robuste."""
    by_pid: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        by_pid[p["participant"]].append(p)

    rows = []
    for pid in sorted(by_pid):
        items = by_pid[pid]
        y_reg = float(items[0]["y_reg"])
        y_pred = float(np.median([x["score"] for x in items]))
        rows.append({
            "participant": pid,
            "y_reg": y_reg,
            "score_median": y_pred,
            "abs_error": abs(y_reg - y_pred),
            "mean_uncertainty": float(np.mean([x["uncertainty"] for x in items])),
        })
    return rows


def wilcoxon_pair(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alternative: str = "greater",
) -> Dict[str, float]:
    """Test de Wilcoxon signé sur paires (ignore les nan)."""
    mask = np.isfinite(values_a) & np.isfinite(values_b)
    a, b = values_a[mask], values_b[mask]
    if len(a) < 5:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(len(a))}
    try:
        stat, p = wilcoxon(a, b, alternative=alternative)
        return {"statistic": float(stat), "p_value": float(p), "n": int(len(a))}
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(len(a))}


def _safe_plots(preds: List[dict], aux: Dict, out_dir: Path) -> None:
    """Graphiques optionnels — ne doit pas faire échouer l'ablation."""
    plot_jobs = (
        ("score_vs_time.png", lambda: plot_score_vs_time(
            aux["curves"], out_dir / "score_vs_time.png", n_trials=len(preds),
        )),
        ("confusion_matrix.png", lambda: plot_confusion(preds, out_dir / "confusion_matrix.png")),
        ("scatter.png", lambda: plot_scatter(preds, out_dir / "scatter.png")),
    )
    for fname, plot_fn in plot_jobs:
        try:
            plot_fn()
        except Exception as exc:
            print(f"  (plot {fname} ignoré : {exc})")


def results_from_preds(
    tag: str,
    preds: List[dict],
    encoder_ckpt: Optional[Path],
) -> Dict:
    fold_rows = per_fold_metrics(preds)
    return {
        "tag": tag,
        "encoder_ckpt": str(encoder_ckpt) if encoder_ckpt else None,
        "global": global_metrics(preds),
        "per_fold": fold_rows,
        "per_participant": per_participant_metrics(preds),
        "r_pearson": [r["r_pearson"] for r in fold_rows],
        "r_spearman": [r["r_spearman"] for r in fold_rows],
        "mae": [r["mae"] for r in fold_rows],
        "mean_uncertainty": [r["mean_uncertainty"] for r in fold_rows],
    }


def load_saved_condition(out_dir: Path, tag: str, encoder_ckpt: Optional[Path]) -> Optional[Dict]:
    """Recharge une condition déjà entraînée (lopo_predictions.pkl)."""
    pred_path = out_dir / tag / LOPO_PREDICTIONS_FILE
    if not pred_path.exists():
        return None
    preds, aux = load_lopo_results(pred_path)
    print(f"\n[Reprise] Condition {tag} rechargée depuis {pred_path} ({len(preds)} prédictions)")
    _safe_plots(preds, aux, out_dir / tag)
    return results_from_preds(tag, preds, encoder_ckpt)


def run_condition(
    tag: str,
    dataset: Dict,
    device: torch.device,
    args: argparse.Namespace,
    encoder_ckpt: Optional[Path],
) -> Dict:
    print("\n" + "=" * 72)
    print(f"  CONDITION {tag} — encoder_ckpt={encoder_ckpt or 'random init'}")
    print("=" * 72)

    out_dir = args.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, aux = run_lopo_mae(
        dataset, device,
        epochs=args.epochs,
        max_folds=args.max_folds,
        patience=args.patience,
        batch_size=args.batch_size,
        mc_passes=args.mc_passes,
        aug_target=args.aug_target,
        no_inline_aug=args.no_inline_aug,
        encoder_ckpt=encoder_ckpt,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
    )

    print_early_stopping_summary(aux.get("fold_train_info", []), args.epochs)
    print_metrics(preds)

    save_lopo_results(out_dir, preds, aux)
    _safe_plots(preds, aux, out_dir)
    return results_from_preds(tag, preds, encoder_ckpt)


def compare_conditions(res_a: Dict, res_b: Dict) -> Dict:
    """Agrège Δr, Wilcoxon et recommandation selon le protocole."""
    r_a = np.array(res_a["r_pearson"], dtype=float)
    r_b = np.array(res_b["r_pearson"], dtype=float)
    mae_a = np.array(res_a["mae"], dtype=float)
    mae_b = np.array(res_b["mae"], dtype=float)

    delta_r = r_a - r_b
    delta_r_mean = float(np.nanmean(delta_r))
    delta_r_std = float(np.nanstd(delta_r))

    err_a = np.array([p["abs_error"] for p in res_a["per_participant"]], dtype=float)
    err_b = np.array([p["abs_error"] for p in res_b["per_participant"]], dtype=float)
    unc_a = np.array([p["mean_uncertainty"] for p in res_a["per_participant"]], dtype=float)
    unc_b = np.array([p["mean_uncertainty"] for p in res_b["per_participant"]], dtype=float)

    wilcoxon_r = wilcoxon_pair(r_a, r_b, alternative="greater")
    wilcoxon_mae = wilcoxon_pair(mae_b, mae_a, alternative="greater")  # A < B → B-A
    wilcoxon_err = wilcoxon_pair(err_b, err_a, alternative="greater")

    if delta_r_mean >= PRACTICAL_DELTA_R and wilcoxon_r["p_value"] < 0.05:
        recommendation = "conserver_atracsys"
    elif delta_r_mean <= -PRACTICAL_DELTA_R:
        recommendation = "retirer_atracsys_investiguer_shift"
    elif abs(delta_r_mean) < 0.02:
        recommendation = "neutre_decision_cout"
    else:
        recommendation = "conserver_prudemment"

    return {
        "delta_r_mean": delta_r_mean,
        "delta_r_std": delta_r_std,
        "delta_r_per_fold": [float(x) for x in delta_r],
        "wilcoxon_r": wilcoxon_r,
        "wilcoxon_mae": wilcoxon_mae,
        "wilcoxon_participant_error": wilcoxon_err,
        "mean_uncertainty_A": float(np.mean(unc_a)),
        "mean_uncertainty_B": float(np.mean(unc_b)),
        "recommendation": recommendation,
        "global_A": res_a["global"],
        "global_B": res_b["global"],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Ablation MAE Atracsys (A=pré-entraîné, B=random) — LOPO 47-fold.",
    )
    ap.add_argument("--data", type=Path, default=Path("data/continuous_per_trial.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/ablation_atracsys"))
    ap.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="best_encoder.pt (MAE Run 8). Défaut : MAE_ENCODER_CKPT env ou ~/icems/results/mae_run8/best_encoder.pt",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-folds", type=int, default=None, help="Limite folds LOPO (test rapide).")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mc-passes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--aug-target", type=int, default=None)
    ap.add_argument("--no-inline-aug", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true", help="Geler l'encodeur MAE (fine-tune GRU seul).")
    ap.add_argument(
        "--condition",
        choices=["both", "A", "B"],
        default="both",
        help="Exécuter les deux conditions ou une seule (reprise).",
    )
    ap.add_argument(
        "--reuse-saved",
        action="store_true",
        help="Recharge A/B depuis lopo_predictions.pkl si présent (évite de ré-entraîner).",
    )
    args = ap.parse_args()

    encoder_path = args.encoder
    if encoder_path is None:
        import os
        env_ckpt = os.environ.get("MAE_ENCODER_CKPT")
        if env_ckpt:
            encoder_path = Path(env_ckpt)
        else:
            encoder_path = (
                Path.home() / "icems" / "results" / "mae_run8_2026-05-25" / "best_encoder.pt"
            )

    if not args.data.exists():
        raise FileNotFoundError(f"{args.data} introuvable.")

    args.out.mkdir(parents=True, exist_ok=True)
    _set_seeds(args.seed)

    with open(args.data, "rb") as f:
        raw = pickle.load(f)
    dataset = step_b._normalize_dataset(raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  ABLATION ATRACSYS — MAE encoder (Solution C) + GRU causal + LOPO")
    print("=" * 72)
    print(f"  Dataset   : {len(dataset)} trials réels")
    print(f"  Device    : {device}")
    print(f"  Encoder A : {encoder_path}")
    print(f"  Seed      : {args.seed}")
    print(f"  Condition : {args.condition}")

    results_path = args.out / "ablation_results.json"
    existing = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            existing = json.load(f)

    res_a, res_b = None, None

    if args.condition in ("both", "A"):
        res_a = load_saved_condition(args.out, "A", encoder_path) if args.reuse_saved else None
        if res_a is None:
            if not encoder_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint MAE introuvable : {encoder_path}\n"
                    f"Passe --encoder ou définis MAE_ENCODER_CKPT."
                )
            res_a = run_condition("A", dataset, device, args, encoder_ckpt=encoder_path)
        existing["A"] = {
            "r_pearson": res_a["r_pearson"],
            "r_spearman": res_a["r_spearman"],
            "mae": res_a["mae"],
            "global": res_a["global"],
            "encoder_ckpt": res_a["encoder_ckpt"],
        }

    if args.condition == "B":
        res_a = load_saved_condition(args.out, "A", encoder_path)
        if res_a is not None:
            existing["A"] = {
                "r_pearson": res_a["r_pearson"],
                "r_spearman": res_a["r_spearman"],
                "mae": res_a["mae"],
                "global": res_a["global"],
                "encoder_ckpt": res_a["encoder_ckpt"],
            }

    if args.condition in ("both", "B"):
        res_b = load_saved_condition(args.out, "B", None) if args.reuse_saved else None
        if res_b is None:
            res_b = run_condition("B", dataset, device, args, encoder_ckpt=None)
        existing["B"] = {
            "r_pearson": res_b["r_pearson"],
            "r_spearman": res_b["r_spearman"],
            "mae": res_b["mae"],
            "global": res_b["global"],
            "encoder_ckpt": None,
        }

    if res_a is None and "A" in existing:
        res_a = existing["A"]
    if res_b is None and "B" in existing:
        res_b = existing["B"]

    if res_a is not None and res_b is not None:
        if isinstance(res_a, dict) and "per_fold" not in res_a:
            # rechargement depuis JSON partiel — comparaison limitée
            comparison = {
                "delta_r_mean": float(
                    np.nanmean(np.array(existing["A"]["r_pearson"]) - np.array(existing["B"]["r_pearson"]))
                ),
                "note": "Comparaison depuis JSON — relancer --condition both pour Wilcoxon complet.",
            }
        else:
            comparison = compare_conditions(res_a, res_b)
            existing["comparison"] = comparison

        print("\n" + "=" * 72)
        print("  SYNTHÈSE ABLATION")
        print("=" * 72)
        if "global_A" in comparison:
            gA, gB = comparison["global_A"], comparison["global_B"]
            print(f"  Global Pearson  A={gA['pearson_r']:+.4f}  B={gB['pearson_r']:+.4f}")
            print(f"  Global Spearman A={gA['spearman_r']:+.4f}  B={gB['spearman_r']:+.4f}")
            print(f"  Global MAE      A={gA['mae']:.4f}  B={gB['mae']:.4f}")
            print(f"  Δr (fold mean)  {comparison['delta_r_mean']:+.4f} ± {comparison['delta_r_std']:.4f}")
            w = comparison["wilcoxon_r"]
            print(f"  Wilcoxon r      p={w['p_value']:.4e}  n={w['n']}")
            print(f"  Recommandation  {comparison['recommendation']}")
        existing["delta_r_mean"] = comparison.get("delta_r_mean")
        existing["wilcoxon_p"] = comparison.get("wilcoxon_r", {}).get("p_value")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"\n✅ Résultats : {results_path.resolve()}")


if __name__ == "__main__":
    main()
