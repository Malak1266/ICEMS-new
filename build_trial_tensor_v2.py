# -*- coding: utf-8 -*-
"""
Construit trial_tensor_v2.pkl depuis data/full_data.json.

Politique v2 (fidélité papier EV-ICEMS) :
  - Masque cinématique = tracking_flag seul (bipolar, scissors, cavitron)
  - Pas de zeroing des cinématiques : valeurs réelles + mask_tracking séparé
  - Distance bip-cav : valeur réelle ; validité via bidist_captured_flag (meta)

Sorties:
  - data/trial_tensor_v2.pkl
  - data/trial_tensor_v2_meta.json

Usage:
    python build_trial_tensor_v2.py
    python build_trial_tensor_v2.py --full_json data/full_data.json
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
    SAMPLE_RATE_HZ,
    SEED,
    TRACKING_FLAG_CHAIN,
    build_trial_record,
    group_by_participant_trial,
    load_full_data,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_trial_tensor_v2")

FEATURE_NAMES = [
    "bip_v", "bip_a", "bip_j",
    "sci_v", "sci_a", "sci_j",
    "cav_v", "cav_a", "cav_j",
    "dist_bip_cav",
]
MASK_NAMES = ["track_bipolar", "track_scissors", "track_cavitron"]


def build_all(full_json: Path) -> list[dict]:
    np.random.seed(SEED)
    raw = load_full_data(full_json)
    by_pt, labels = group_by_participant_trial(raw)

    records: list[dict] = []
    skipped: list[str] = []

    for (pid, tid) in sorted(by_pt.keys(), key=lambda x: (x[0], x[1])):
        rec = build_trial_record(
            pid,
            tid,
            by_pt[(pid, tid)],
            labels,
            kinematic_flag_chain=TRACKING_FLAG_CHAIN,
            zero_kinematics=False,
            zero_distance=False,
            mask_field="mask_tracking",
        )
        if rec is None:
            skipped.append(f"{pid}/{tid}")
            continue
        records.append(rec)

    log.info("Trials construits (v2) : %d", len(records))
    if skipped:
        log.warning("Trials ignorés (T=0) : %s", skipped)

    return records


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
            "kinematic_mask": "tracking_flag (no fallback to inuse_flag or captured_flag)",
            "feature_zeroing": "none — real kinematics preserved; validity in mask_tracking (T,3)",
            "distance_mask": "bidist_captured_flag documented separately; distance not zeroed",
            "alignment": "T = min longueur commune sur tous les instruments/métriques du trial",
        },
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info("Sauvegardé : %s", out_pkl)
    log.info("Sauvegardé : %s", out_meta)


def run(full_json: Path, out_pkl: Path, out_meta: Path) -> int:
    log.info("Construction v2 depuis %s (seed=%d)", full_json, SEED)
    records = build_all(full_json)
    save_outputs(records, out_pkl, out_meta)

    participants = {r["participant"] for r in records}
    log.info("Sortie : %d participants, %d trials", len(participants), len(records))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full_json", type=Path, default=Path("data/full_data.json"))
    ap.add_argument("--out_pkl", type=Path, default=Path("data/trial_tensor_v2.pkl"))
    ap.add_argument("--out_meta", type=Path, default=Path("data/trial_tensor_v2_meta.json"))
    args = ap.parse_args()
    raise SystemExit(run(args.full_json, args.out_pkl, args.out_meta))


if __name__ == "__main__":
    main()
