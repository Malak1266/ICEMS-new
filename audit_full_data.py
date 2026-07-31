# -*- coding: utf-8 -*-
"""
Audit de couverture de data/full_data.json avant construction du tenseur trial-level.

Usage:
    python audit_full_data.py
    python audit_full_data.py --full_json data/full_data.json --out_csv results/audit_full_data.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from full_data_io import (  # noqa: E402
    CAPTURE_FLAG_CHAIN,
    DIST_FLAG_CHAIN,
    DIST_INSTRUMENT,
    MONO_INSTRUMENTS,
    MONO_REQUIRED_METRICS,
    capture_fraction,
    get_flag_bundle,
    group_by_participant_trial,
    load_full_data,
    metric_present,
    metric_length,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("audit_full_data")


def _audit_trial(inst_map: dict[str, dict]) -> dict:
    row: dict = {}
    missing: list[str] = []

    for inst in MONO_INSTRUMENTS:
        bundle = inst_map.get(inst)
        inst_ok = bundle is not None
        for met in MONO_REQUIRED_METRICS:
            present = inst_ok and metric_present(bundle, met)
            row[f"{inst}_{met}_present"] = int(present)
            if not present:
                missing.append(f"{inst}.{met}")

        cap_present = inst_ok and get_flag_bundle(bundle, CAPTURE_FLAG_CHAIN) is not None
        row[f"{inst}_capture_flag_present"] = int(cap_present)
        if not cap_present:
            missing.append(f"{inst}.capture_flag")

        row[f"T_{inst}"] = metric_length(bundle, "velocity") if bundle else 0
        frac = capture_fraction(bundle) if bundle else None
        row[f"{inst}_captured_frac"] = "" if frac is None else round(frac, 6)

    dist_bundle = inst_map.get(DIST_INSTRUMENT)
    dist_present = dist_bundle is not None and metric_present(dist_bundle, "distance")
    row["bipolar_to_cavitron_distance_present"] = int(dist_present)
    if not dist_present:
        missing.append(f"{DIST_INSTRUMENT}.distance")

    dist_flag_present = (
        dist_bundle is not None
        and get_flag_bundle(dist_bundle, DIST_FLAG_CHAIN) is not None
    )
    row["bipolar_to_cavitron_flag_present"] = int(dist_flag_present)
    if not dist_flag_present:
        missing.append(f"{DIST_INSTRUMENT}.bidist_captured_flag")

    row["T_dist_bipolar_to_cavitron"] = (
        metric_length(dist_bundle, "distance") if dist_bundle else 0
    )
    dist_frac = capture_fraction(dist_bundle) if dist_bundle else None
    if dist_bundle is not None:
        flag = get_flag_bundle(dist_bundle, DIST_FLAG_CHAIN)
        if flag is not None and len(flag) > 0:
            dist_frac = float(np_mean_bool(flag))
    row["dist_bipolar_to_cavitron_captured_frac"] = (
        "" if dist_frac is None else round(dist_frac, 6)
    )

    row["complete"] = int(len(missing) == 0)
    row["missing_items"] = ";".join(missing)
    return row


def np_mean_bool(flag) -> float:
    import numpy as np

    return float(np.mean(np.asarray(flag, dtype=bool)))


def run_audit(full_json: Path, out_csv: Path) -> int:
    log.info("Chargement de %s", full_json)
    raw = load_full_data(full_json)
    by_pt, _ = group_by_participant_trial(raw)

    participants = sorted({pid for pid, _ in by_pt})
    trials = sorted(by_pt.keys(), key=lambda x: (x[0], x[1]))

    log.info("Participants couverts : %d", len(participants))
    log.info("Trials couverts       : %d", len(trials))

    fieldnames = [
        "participant",
        "trial",
        "complete",
        "missing_items",
    ]
    for inst in MONO_INSTRUMENTS:
        for met in MONO_REQUIRED_METRICS:
            fieldnames.append(f"{inst}_{met}_present")
        fieldnames.extend([
            f"{inst}_capture_flag_present",
            f"T_{inst}",
            f"{inst}_captured_frac",
        ])
    fieldnames.extend([
        "bipolar_to_cavitron_distance_present",
        "bipolar_to_cavitron_flag_present",
        "T_dist_bipolar_to_cavitron",
        "dist_bipolar_to_cavitron_captured_frac",
    ])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    incomplete: list[str] = []
    complete_count = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid, tid in trials:
            audit = _audit_trial(by_pt[(pid, tid)])
            if audit["complete"]:
                complete_count += 1
            else:
                incomplete.append(f"{pid}/{tid} ({audit['missing_items']})")

            row = {"participant": pid, "trial": tid, **audit}
            writer.writerow(row)

    log.info("Trials complets       : %d / %d", complete_count, len(trials))
    log.info("Rapport CSV           : %s", out_csv)

    if incomplete:
        log.warning("Trials incomplets (%d) :", len(incomplete))
        for item in incomplete:
            log.warning("  - %s", item)
    else:
        log.info("Aucun trial incomplet.")

    target_ok = len(participants) == 47 and len(trials) == 136
    if target_ok:
        log.info("OK : objectif 47 participants / 136 trials atteint.")
    else:
        log.warning(
            "ATTENTION : attendu 47 participants / 136 trials, obtenu %d / %d.",
            len(participants),
            len(trials),
        )

    return 0 if target_ok and not incomplete else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full_json", type=Path, default=Path("data/full_data.json"))
    ap.add_argument("--out_csv", type=Path, default=Path("results/audit_full_data.csv"))
    args = ap.parse_args()
    raise SystemExit(run_audit(args.full_json, args.out_csv))


if __name__ == "__main__":
    main()
