# -*- coding: utf-8 -*-
"""
Mesures de vérification v1 vs v2 — aucun entraînement.

Sorties : results/preproc_v2/
  - detection_rates.json
  - anova_by_group.json
  - nonzero_fractions.json
  - sanity_correlations.json
  - counts.json
  - v1_vs_v2_summary.md

Usage:
    python preproc_v2_verify.py
    python preproc_v2_verify.py --v2_pkl data/trial_tensor_v2.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from full_data_io import (  # noqa: E402
    CAPTURE_FLAG_CHAIN,
    MONO_INSTRUMENTS,
    MONO_KIN_METRICS,
    TRACKING_FLAG_CHAIN,
    _as_1d,
    _first,
    align_len,
    build_trial_record,
    get_flag_bundle,
    group_by_participant_trial,
    load_full_data,
    raw_metric,
    trial_common_length,
    trial_labels,
)

PAPER_RATES = {"bipolar": 0.88, "scissors": 0.79, "cavitron": 0.74}
GROUPS = ("novice", "junior", "senior", "expert")
INST_SHORT = {"bipolar": "bip", "scissors": "sci", "cavitron": "cav"}
KIN_COLS = {
    "bipolar": (0, 1, 2),
    "scissors": (3, 4, 5),
    "cavitron": (6, 7, 8),
}
OUT_DIR = Path("results/preproc_v2")
KIN_EPS = 1e-9


def _load_v1_records(full_json: Path, v1_pkl: Path | None) -> list[dict]:
    if v1_pkl is not None and v1_pkl.exists():
        with open(v1_pkl, "rb") as f:
            return pickle.load(f)
    raw = load_full_data(full_json)
    by_pt, labels = group_by_participant_trial(raw)
    records = []
    for (pid, tid) in sorted(by_pt.keys(), key=lambda x: (x[0], x[1])):
        rec = build_trial_record(
            pid, tid, by_pt[(pid, tid)], labels,
            kinematic_flag_chain=CAPTURE_FLAG_CHAIN,
            zero_kinematics=True,
            zero_distance=True,
            mask_field="mask",
        )
        if rec is not None:
            records.append(rec)
    return records


def _trial_mask(rec: dict) -> np.ndarray:
    if "mask_tracking" in rec:
        return rec["mask_tracking"]
    return rec["mask"]


def detection_rates_from_json(full_json: Path, flag_chain: tuple[str, ...]) -> dict:
    raw = load_full_data(full_json)
    by_pt, labels = group_by_participant_trial(raw)
    rates: dict[str, list[float]] = {inst: [] for inst in MONO_INSTRUMENTS}
    groups: dict[str, list[str]] = {inst: [] for inst in MONO_INSTRUMENTS}

    for (pid, tid), inst_map in sorted(by_pt.items(), key=lambda x: (x[0][0], x[0][1])):
        _, _, group4, *_ = trial_labels(pid, tid, inst_map, labels)
        g = group4 if group4 in GROUPS else "unknown"
        t = trial_common_length(inst_map, flag_chain)
        if t <= 0:
            continue
        for inst in MONO_INSTRUMENTS:
            bundle = inst_map.get(inst)
            if bundle is None:
                continue
            flag = get_flag_bundle(bundle, flag_chain)
            if flag is None:
                continue
            aligned, _ = align_len(flag)
            flag = np.asarray(aligned[0], dtype=bool)[:t]
            rates[inst].append(float(np.mean(flag)))
            groups[inst].append(g)

    return {"rates": rates, "groups": groups}


def detection_rates_from_pkl(records: list[dict]) -> dict:
    rates: dict[str, list[float]] = {inst: [] for inst in MONO_INSTRUMENTS}
    groups: dict[str, list[str]] = {inst: [] for inst in MONO_INSTRUMENTS}
    for rec in records:
        g = rec.get("group4") or "unknown"
        mask = _trial_mask(rec)
        for i, inst in enumerate(MONO_INSTRUMENTS):
            rates[inst].append(float(mask[:, i].mean()))
            groups[inst].append(g)
    return {"rates": rates, "groups": groups}


def anova_by_group(rates: list[float], groups: list[str]) -> float:
    by_g = {g: [] for g in GROUPS}
    for r, g in zip(rates, groups):
        if g in by_g:
            by_g[g].append(r)
    samples = [by_g[g] for g in GROUPS if by_g[g]]
    if len(samples) < 2:
        return float("nan")
    return float(stats.f_oneway(*samples).pvalue)


def nonzero_fraction(records: list[dict], version: str) -> dict[str, float]:
    """Fraction de frames avec au moins un canal cinématique non nul par instrument."""
    sums = {inst: [] for inst in MONO_INSTRUMENTS}
    for rec in records:
        x = rec["X_feat"]
        for inst, cols in KIN_COLS.items():
            block = x[:, cols]
            nz = np.any(np.abs(block) > KIN_EPS, axis=1)
            sums[inst].append(float(np.mean(nz)))
    return {inst: float(np.mean(sums[inst])) for inst in MONO_INSTRUMENTS}


def sanity_check_v2(records_v2: list[dict], full_json: Path) -> dict:
    raw = load_full_data(full_json)
    by_pt, _ = group_by_participant_trial(raw)

    corrs: list[float] = []
    n_checked_false = 0
    n_nonzero_when_false = 0
    examples_false_nonzero: list[str] = []

    for rec in records_v2:
        key = (rec["participant"], rec["trial"])
        inst_map = by_pt.get(key)
        if inst_map is None:
            continue
        t = rec["T"]
        mask = rec["mask_tracking"]
        x = rec["X_feat"]

        for i_inst, inst in enumerate(MONO_INSTRUMENTS):
            bundle = inst_map.get(inst)
            if bundle is None:
                continue
            trk = get_flag_bundle(bundle, TRACKING_FLAG_CHAIN)
            if trk is None:
                continue
            aligned, _ = align_len(trk)
            trk = np.asarray(aligned[0], dtype=bool)[:t]
            cols = KIN_COLS[inst]

            for col, met in zip(cols, MONO_KIN_METRICS):
                raw_vals = raw_metric(bundle, met, t, float)
                sel = trk
                if sel.sum() >= 2:
                    a = x[sel, col]
                    b = raw_vals[sel]
                    if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                        corrs.append(float(np.corrcoef(a, b)[0, 1]))

            false_mask = ~trk
            n_checked_false += int(false_mask.sum())
            if false_mask.any():
                block = x[false_mask][:, list(cols)]
                has_nz = np.any(np.abs(block) > KIN_EPS, axis=1)
                n_nonzero_when_false += int(has_nz.sum())
                if has_nz.any() and len(examples_false_nonzero) < 5:
                    idx = np.where(false_mask)[0][np.where(has_nz)[0][0]]
                    examples_false_nonzero.append(
                        f"{key[0]}/{key[1]} {inst} frame={idx}"
                    )

    return {
        "corr_tracking_true_mean": float(np.mean(corrs)) if corrs else None,
        "corr_tracking_true_min": float(np.min(corrs)) if corrs else None,
        "corr_tracking_true_max": float(np.max(corrs)) if corrs else None,
        "n_frames_tracking_false": n_checked_false,
        "n_nonzero_kin_when_tracking_false": n_nonzero_when_false,
        "frac_nonzero_when_tracking_false": (
            n_nonzero_when_false / n_checked_false if n_checked_false else None
        ),
        "examples_false_nonzero": examples_false_nonzero,
        "zeroing_stopped": n_nonzero_when_false > 0,
    }


def counts(records: list[dict]) -> dict:
    return {
        "n_participants": len({r["participant"] for r in records}),
        "n_trials": len(records),
    }


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * x:.1f}%"


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def write_summary(
    out_dir: Path,
    det_v1: dict,
    det_v2: dict,
    anova_v1: dict,
    anova_v2: dict,
    nz_v1: dict,
    nz_v2: dict,
    sanity: dict,
    cnt_v2: dict,
) -> None:
    lines = [
        "# Prétraitement v1 vs v2 — résumé des mesures",
        "",
        "**Politique v2 :** `tracking_flag` seul, pas de zeroing, `mask_tracking` exposé séparément.",
        "",
        "## Commandes",
        "",
        "```bash",
        "python build_trial_tensor_v2.py",
        "python preproc_v2_verify.py",
        "```",
        "",
        "## (a) Taux de détection par instrument (moyenne trial-level)",
        "",
        "| Instrument | Papier | v1 (captured_flag) | v2 (tracking_flag) |",
        "|------------|--------|--------------------|----------------------|",
    ]
    for inst in MONO_INSTRUMENTS:
        m1 = float(np.mean(det_v1["rates"][inst]))
        m2 = float(np.mean(det_v2["rates"][inst]))
        lines.append(
            f"| {inst} | {_fmt_pct(PAPER_RATES[inst])} | {_fmt_pct(m1)} | {_fmt_pct(m2)} |"
        )

    lines += [
        "",
        "## (b) ANOVA taux ~ groupe (p-value par instrument)",
        "",
        "| Instrument | v1 p | v2 p |",
        "|------------|------|------|",
    ]
    for inst in MONO_INSTRUMENTS:
        lines.append(
            f"| {inst} | {_fmt_p(anova_v1[inst])} | {_fmt_p(anova_v2[inst])} |"
        )

    lines += [
        "",
        "## (c) Fraction de frames avec cinématique non nulle",
        "",
        "| Instrument | v1 | v2 |",
        "|------------|----|----|",
    ]
    for inst in MONO_INSTRUMENTS:
        lines.append(
            f"| {inst} | {_fmt_pct(nz_v1[inst])} | {_fmt_pct(nz_v2[inst])} |"
        )

    lines += [
        "",
        "## (d) Sanity v2",
        "",
        f"- Corrélation cinématiques v2 vs `full_data.json` (tracking_flag=True) : "
        f"moyenne={sanity['corr_tracking_true_mean']:.6f}, "
        f"min={sanity['corr_tracking_true_min']:.6f}",
        f"- Frames tracking_flag=False avec cinématique ≠ 0 : "
        f"{sanity['n_nonzero_kin_when_tracking_false']}/{sanity['n_frames_tracking_false']} "
        f"({_fmt_pct(sanity['frac_nonzero_when_tracking_false'])})",
        f"- Zeroing arrêté : **{'oui' if sanity['zeroing_stopped'] else 'non'}**",
        "",
        "## (e) Comptage final (v2)",
        "",
        f"- Participants : **{cnt_v2['n_participants']}**",
        f"- Trials : **{cnt_v2['n_trials']}**",
        "",
        "## Verdict",
        "",
    ]

    p_improved = sum(
        1 for inst in MONO_INSTRUMENTS
        if (not np.isnan(anova_v1[inst]) and not np.isnan(anova_v2[inst])
            and anova_v2[inst] > anova_v1[inst])
    )
    closer_paper = sum(
        1 for inst in MONO_INSTRUMENTS
        if abs(float(np.mean(det_v2["rates"][inst])) - PAPER_RATES[inst])
        < abs(float(np.mean(det_v1["rates"][inst])) - PAPER_RATES[inst])
    )
    ok_count = cnt_v2["n_participants"] == 47 and cnt_v2["n_trials"] == 136

    lines.append(
        f"1. v2 rapproche les taux de détection du papier sur {closer_paper}/3 instruments "
        f"(tracking_flag vs captured_flag)."
    )
    lines.append(
        f"2. Le biais inter-groupes (ANOVA) diminue sur {p_improved}/3 instruments "
        f"sous v2 ; objectif papier p>0.05 : "
        + ", ".join(
            f"{inst} p={_fmt_p(anova_v2[inst])}" for inst in MONO_INSTRUMENTS
        )
        + "."
    )
    lines.append(
        f"3. Cohorte conservée ({cnt_v2['n_participants']}/{cnt_v2['n_trials']}) "
        f"{'✓' if ok_count else 'ATTENTION'} ; cinématiques réelles non zéroées avec masque séparé."
    )

    (out_dir / "v1_vs_v2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full_json", type=Path, default=Path("data/full_data.json"))
    ap.add_argument("--v1_pkl", type=Path, default=Path("data/trial_tensor_v2.pkl"))
    ap.add_argument("--v2_pkl", type=Path, default=Path("data/trial_tensor_v2.pkl"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v1_records = _load_v1_records(args.full_json, args.v1_pkl)
    if not args.v2_pkl.exists():
        raise SystemExit(f"v2 manquant : {args.v2_pkl} — lancer build_trial_tensor_v2.py")
    with open(args.v2_pkl, "rb") as f:
        v2_records = pickle.load(f)

    det_v1 = detection_rates_from_pkl(v1_records)
    det_v2 = detection_rates_from_pkl(v2_records)

    anova_v1 = {
        inst: anova_by_group(det_v1["rates"][inst], det_v1["groups"][inst])
        for inst in MONO_INSTRUMENTS
    }
    anova_v2 = {
        inst: anova_by_group(det_v2["rates"][inst], det_v2["groups"][inst])
        for inst in MONO_INSTRUMENTS
    }

    nz_v1 = nonzero_fraction(v1_records, "v1")
    nz_v2 = nonzero_fraction(v2_records, "v2")
    sanity = sanity_check_v2(v2_records, args.full_json)
    cnt_v2 = counts(v2_records)

    det_table = {
        "paper": PAPER_RATES,
        "v1": {inst: float(np.mean(det_v1["rates"][inst])) for inst in MONO_INSTRUMENTS},
        "v2": {inst: float(np.mean(det_v2["rates"][inst])) for inst in MONO_INSTRUMENTS},
    }
    with open(OUT_DIR / "detection_rates.json", "w", encoding="utf-8") as f:
        json.dump(det_table, f, indent=2)

    with open(OUT_DIR / "anova_by_group.json", "w", encoding="utf-8") as f:
        json.dump({"v1": anova_v1, "v2": anova_v2}, f, indent=2)

    with open(OUT_DIR / "nonzero_fractions.json", "w", encoding="utf-8") as f:
        json.dump({"v1": nz_v1, "v2": nz_v2}, f, indent=2)

    with open(OUT_DIR / "sanity_correlations.json", "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    with open(OUT_DIR / "counts.json", "w", encoding="utf-8") as f:
        json.dump(cnt_v2, f, indent=2)

    write_summary(OUT_DIR, det_v1, det_v2, anova_v1, anova_v2, nz_v1, nz_v2, sanity, cnt_v2)
    print(f"Écrit : {OUT_DIR / 'v1_vs_v2_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
