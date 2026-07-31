# -*- coding: utf-8 -*-
"""
Construit le tenseur trial-level depuis data/full_data.json.

Sorties:
  - data/trial_tensor_v2.pkl  (list[dict])
  - data/trial_tensor_v2_meta.json

Usage:
    python build_trial_tensor.py
    python build_trial_tensor.py --full_json data/full_data.json --skip_validation
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from full_data_io import (  # noqa: E402
    CAPTURE_FLAG_CHAIN,
    MONO_INSTRUMENTS,
    SAMPLE_RATE_HZ,
    SEED,
    build_trial_record,
    group_by_participant_trial,
    load_full_data,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_trial_tensor")

FEATURE_NAMES = [
    "bip_v", "bip_a", "bip_j",
    "sci_v", "sci_a", "sci_j",
    "cav_v", "cav_a", "cav_j",
    "dist_bip_cav",
]
MASK_NAMES = ["cap_bipolar", "cap_scissors", "cap_cavitron"]


def build_one_trial(
    pid: str,
    tid: str,
    inst_map: dict[str, dict],
    labels: dict,
) -> dict | None:
    return build_trial_record(
        pid,
        tid,
        inst_map,
        labels,
        kinematic_flag_chain=CAPTURE_FLAG_CHAIN,
        zero_kinematics=True,
        zero_distance=True,
        mask_field="mask",
    )


def build_all(full_json: Path) -> list[dict]:
    np.random.seed(SEED)
    raw = load_full_data(full_json)
    by_pt, labels = group_by_participant_trial(raw)

    records: list[dict] = []
    skipped: list[str] = []

    for (pid, tid) in sorted(by_pt.keys(), key=lambda x: (x[0], x[1])):
        rec = build_one_trial(pid, tid, by_pt[(pid, tid)], labels)
        if rec is None:
            skipped.append(f"{pid}/{tid}")
            continue
        records.append(rec)

    log.info("Trials construits : %d", len(records))
    if skipped:
        log.warning("Trials ignorés (T=0) : %s", skipped)

    return records


def _corr_channel(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def validate_vs_continuous(records: list[dict], pkl_path: Path) -> None:
    if not pkl_path.exists():
        log.warning("Validation (a) ignorée : %s introuvable.", pkl_path)
        return

    with open(pkl_path, "rb") as f:
        ref = pickle.load(f)

    corrs: list[float] = []
    mismatches: list[str] = []

    for rec in records:
        key = (rec["participant"], rec["trial"])
        if key not in ref:
            mismatches.append(f"{key[0]}/{key[1]} absent du pkl de référence")
            continue

        x_new = rec["X_feat"][:, :9]
        x_ref = np.asarray(ref[key]["X"][:, :9], dtype=np.float32)
        t = min(len(x_new), len(x_ref))
        for ch in range(9):
            c = _corr_channel(x_new[:t, ch], x_ref[:t, ch])
            if not np.isnan(c):
                corrs.append(c)
            if not np.isnan(c) and c < 0.999:
                mismatches.append(
                    f"{key[0]}/{key[1]} canal {FEATURE_NAMES[ch]} corr={c:.6f}"
                )

    if corrs:
        log.info(
            "Validation (a) corrélation 9 canaux cinématiques vs %s : "
            "moyenne=%.6f min=%.6f max=%.6f",
            pkl_path.name,
            float(np.mean(corrs)),
            float(np.min(corrs)),
            float(np.max(corrs)),
        )
    if mismatches:
        log.warning(
            "Validation (a) ALERTE : %d paires canal/trial avec corr < 0.999 :",
            len(mismatches),
        )
        for m in mismatches[:20]:
            log.warning("  - %s", m)
        if len(mismatches) > 20:
            log.warning("  ... et %d autres", len(mismatches) - 20)
    else:
        log.info("Validation (a) OK : corrélation ~1.0 sur les 9 canaux cinématiques.")


def validate_mask_by_group(records: list[dict]) -> None:
    by_group: dict[str, list[dict]] = {}
    for rec in records:
        g = rec.get("group4") or "unknown"
        by_group.setdefault(g, []).append(rec)

    log.info("Validation (b) fraction moyenne de frames captées par groupe :")
    ge1_means: dict[str, float] = {}
    ge2_means: dict[str, float] = {}

    for group, items in sorted(by_group.items()):
        ge1 = [float(rec["mask_ge1"].mean()) for rec in items]
        ge2 = [float(rec["mask_ge2"].mean()) for rec in items]
        ge1_means[group] = float(np.mean(ge1))
        ge2_means[group] = float(np.mean(ge2))
        log.info(
            "  %-8s : ge1=%.4f  ge2=%.4f  (n=%d trials)",
            group,
            ge1_means[group],
            ge2_means[group],
            len(items),
        )

    if len(ge1_means) >= 2:
        spread_ge1 = max(ge1_means.values()) - min(ge1_means.values())
        spread_ge2 = max(ge2_means.values()) - min(ge2_means.values())
        if spread_ge1 > 0.15 or spread_ge2 > 0.15:
            log.warning(
                "WARNING : le masque pourrait corréler avec le label "
                "(écart ge1=%.3f, ge2=%.3f entre groupes d'expertise).",
                spread_ge1,
                spread_ge2,
            )


def validate_t_by_group(records: list[dict]) -> None:
    by_group: dict[str, list[int]] = {}
    short_ge2: list[str] = []

    for rec in records:
        g = rec.get("group4") or "unknown"
        by_group.setdefault(g, []).append(rec["T"])
        n_ge2 = int(rec["mask_ge2"].sum())
        if n_ge2 < 50:
            short_ge2.append(
                f"{rec['participant']}/{rec['trial']} (group={g}, ge2_frames={n_ge2}, T={rec['T']})"
            )

    log.info("Validation (c) distribution de T par groupe :")
    for group, ts in sorted(by_group.items()):
        arr = np.asarray(ts)
        log.info(
            "  %-8s : min=%d  médiane=%d  max=%d  (n=%d)",
            group,
            int(arr.min()),
            int(np.median(arr)),
            int(arr.max()),
            len(arr),
        )

    if short_ge2:
        log.warning(
            "Validation (c) %d trials avec <50 frames sous ge2 :",
            len(short_ge2),
        )
        for item in short_ge2[:15]:
            log.warning("  - %s", item)
        if len(short_ge2) > 15:
            log.warning("  ... et %d autres", len(short_ge2) - 15)


def validate_counts(records: list[dict]) -> None:
    participants = {r["participant"] for r in records}
    n_part = len(participants)
    n_trials = len(records)
    log.info("Validation (d) sortie : %d participants, %d trials", n_part, n_trials)
    if n_part == 47 and n_trials == 136:
        log.info("Validation (d) OK : 47 participants / 136 trials.")
    else:
        log.warning(
            "Validation (d) ATTENTION : attendu 47/136, obtenu %d/%d.",
            n_part,
            n_trials,
        )


def save_outputs(records: list[dict], out_pkl: Path, out_meta: Path) -> None:
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(records, f)

    meta = {
        "seed": SEED,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "feature_names": FEATURE_NAMES,
        "mask_names": MASK_NAMES,
        "n_trials": len(records),
        "n_participants": len({r["participant"] for r in records}),
        "policy": {
            "kinematic_mask": "captured_flag (fallback inuse_flag -> tracking_flag)",
            "feature_zeroing": "captured_flag=False => feature=0 (mask conservé séparément)",
            "distance_mask": "bidist_captured_flag (fallback captured_flag)",
            "alignment": "T = min longueur commune sur tous les instruments/métriques du trial",
        },
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info("Sauvegardé : %s", out_pkl)
    log.info("Sauvegardé : %s", out_meta)


def run(
    full_json: Path,
    out_pkl: Path,
    out_meta: Path,
    ref_pkl: Path,
    skip_validation: bool,
) -> int:
    log.info("Construction depuis %s (seed=%d)", full_json, SEED)
    records = build_all(full_json)
    save_outputs(records, out_pkl, out_meta)

    if not skip_validation:
        log.info("--- Sanity checks (non bloquants) ---")
        validate_vs_continuous(records, ref_pkl)
        validate_mask_by_group(records)
        validate_t_by_group(records)
        validate_counts(records)

    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full_json", type=Path, default=Path("data/full_data.json"))
    ap.add_argument("--out_pkl", type=Path, default=Path("data/trial_tensor_v2.pkl"))
    ap.add_argument("--out_meta", type=Path, default=Path("data/trial_tensor_v2_meta.json"))
    ap.add_argument(
        "--ref_pkl",
        type=Path,
        default=Path("data/continuous_per_trial.pkl"),
        help="Référence pour corrélation des 9 canaux cinématiques",
    )
    ap.add_argument("--skip_validation", action="store_true")
    args = ap.parse_args()

    out_meta = args.out_meta
    if out_meta is None:
        out_meta = args.out_pkl.with_suffix("").with_name(args.out_pkl.stem + "_meta.json")

    raise SystemExit(
        run(args.full_json, args.out_pkl, out_meta, args.ref_pkl, args.skip_validation)
    )


if __name__ == "__main__":
    main()
