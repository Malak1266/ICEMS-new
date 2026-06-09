# -*- coding: utf-8 -*-
"""
Generate dataset (Format A: list[dict]) from full_data.json for ICEMS.
Option B: entries for mono-instruments AND distance-only pseudo-instruments.

Comportement clé:
- Mono (bipolar/scissors/cavitron):
  * Requiert: timestamp, position, captured_flag
  * Masquage strict par captured_flag (False -> 0)
  * velocity/acceleration/jerk: si absents -> zeros (aucune dérivation)
  * distances (bipolar–cavitron / bipolar–scissors) injectées si disponibles, masquées par bidist_captured_flag (fallback captured_flag)
- Bimanual (bipolar_to_cavitron / bipolar_to_scissors) => "distance-only":
  * N'exige PAS position/captured_flag
  * Requiert: distance (+ bidist_captured_flag ou fallback captured_flag pour le masquage)
  * timestamp pris sur bipolar du même (participant, trial) si dispo; sinon synthétique à 10 Hz
  * Toutes les autres features mono (pos/vel/acc/jerk) = zeros

Sorties:
- PKL: list[dict] avec .data = [metrics x T]
- META: noms des métriques & label maps
"""

import os
import json
import pickle
import argparse
import numpy as np
from collections import defaultdict

SAMPLE_RATE_HZ = 10.0

# --------------------- utils ---------------------
def _first(bundle, key):
    if bundle is None:
        return None
    arrs = bundle.get(key, None)
    if arrs is None:
        return None
    if isinstance(arrs, list) and arrs:
        return arrs[0]
    return arrs

def _as_1d(x, dtype=float):
    if x is None:
        return None
    a = np.asarray(x, dtype=dtype).ravel()
    return a

def _as_pos_3xN(pos):
    P = np.asarray(pos, dtype=float)
    if P.ndim != 2:
        raise ValueError("position must be 2D")
    if P.shape[0] == 3:
        return P
    if P.shape[1] == 3:
        return P.T
    raise ValueError(f"position not 3D compatible: {P.shape}")

def _align_len(*arrs):
    Ls = [len(a) for a in arrs if a is not None]
    if not Ls:
        return [None for _ in arrs], 0
    N = min(Ls)
    if N <= 0:
        return [None for _ in arrs], 0
    out = []
    for a in arrs:
        if a is None:
            out.append(None)
        else:
            out.append(a[:N])
    return out, N

def _pad_or_cut(a, N, dtype=float):
    a = np.asarray(a, dtype=dtype).ravel()
    if len(a) >= N:
        return a[:N]
    out = np.zeros(N, dtype=dtype)
    out[:len(a)] = a
    return out

def _zeros(N):
    return np.zeros(N, dtype=float)

# --------------------- core build ---------------------
def build_dataset(full_json, out_pkl, out_meta):
    print(f"🔄 Loading: {full_json}")
    with open(full_json, "r", encoding="utf-8") as f:
        D = json.load(f)

    req = ["participant","trial","instrument","metric","data","expertise","level"]
    for k in req:
        if k not in D:
            raise KeyError(f"Missing key in full_data: {k}")

    ids = list(D["data"].keys())
    by_combo = defaultdict(dict)
    labels_by_combo = {}

    for i in ids:
        pid  = str(D["participant"][i])
        tr   = str(D["trial"][i])
        inst = str(D["instrument"][i])
        met  = str(D["metric"][i])
        arr  = D["data"][i]
        key  = (pid, tr, inst)

        labels_by_combo[key] = ((D["expertise"][i] or "").strip(),
                                (D["level"][i] or "").strip())

        by_combo[key].setdefault(met, []).append(arr)

    all_exp = sorted({labels_by_combo[k][0] for k in labels_by_combo})
    all_lvl = sorted({labels_by_combo[k][1] for k in labels_by_combo})
    label_map_expertise = {e:i for i,e in enumerate(all_exp)}
    label_map_level     = {e:i for i,e in enumerate(all_lvl)}
    print("🧭 expertise map:", label_map_expertise)
    print("🧭 level map    :", label_map_level)

    by_pt = defaultdict(dict)
    for (pid,tr,inst), bundle in by_combo.items():
        by_pt[(pid,tr)][inst] = bundle

    entries = []
    kept = skipped = 0
    skip_reasons = defaultdict(int)

    for (pid,tr), inst_map in by_pt.items():

        def read_dist(inst_name, N_target):
            b = inst_map.get(inst_name, None)
            if b is None:
                return None
            d  = _first(b, "distance")
            if d is None:
                return None
            d = _as_1d(d, float)
            m  = _first(b, "bidist_captured_flag")
            if m is None:
                m = _first(b, "captured_flag")
            if m is None:
                return None
            m = _as_1d(m, bool)
            aligned, L = _align_len(d, m)
            if L == 0:
                return None
            d_al, m_al = aligned
            d_final = _pad_or_cut(d_al, N_target, float)
            m_final = _pad_or_cut(m_al, N_target, bool)
            return np.where(m_final, d_final, 0.0)

        for inst, B in inst_map.items():
            exp_str, lvl_str = labels_by_combo[(pid,tr,inst)]
            y_exp = int(label_map_expertise.get(exp_str, 0))
            y_lvl = int(label_map_level.get(lvl_str, 0))

            # ---------- BIMANUAL: distance-only ----------
            if inst in ("bipolar_to_cavitron", "bipolar_to_scissors"):
                dist = _first(B, "distance")
                bid  = _first(B, "bidist_captured_flag")
                capb = _first(B, "captured_flag")

                if dist is None:
                    skipped += 1
                    skip_reasons["bimanual_no_distance"] += 1
                    continue

                dist = _as_1d(dist, float)
                L = len(dist)
                if bid is not None:
                    mask = _as_1d(bid, bool)[:L]
                elif capb is not None:
                    mask = _as_1d(capb, bool)[:L]
                else:
                    mask = np.ones(L, dtype=bool)

                Ts_ref = None
                if "bipolar" in inst_map:
                    Ts_ref = _first(inst_map["bipolar"], "timestamp")
                    if Ts_ref is not None:
                        Ts_ref = _as_1d(Ts_ref, float)
                        Ts_ref = _pad_or_cut(Ts_ref, L, float)
                if Ts_ref is None:
                    Ts_ref = np.arange(L, dtype=float) / SAMPLE_RATE_HZ

                Xp = _zeros(L); Yp = _zeros(L); Zp = _zeros(L)
                pos_mag = _zeros(L)
                vel = _zeros(L); acc = _zeros(L); jerk = _zeros(L)

                if inst == "bipolar_to_cavitron":
                    dist_bip_cav = np.where(mask, dist, 0.0)
                    dist_bip_sci = _zeros(L)
                else:
                    dist_bip_cav = _zeros(L)
                    dist_bip_sci = np.where(mask, dist, 0.0)

                rows = [pos_mag, vel, acc, jerk, dist_bip_cav, dist_bip_sci]

                label_row_exp = np.full(L, y_exp, dtype=float)
                label_row_lvl = np.full(L, y_lvl, dtype=float)
                M = np.vstack([label_row_exp, label_row_lvl] + [np.asarray(r, dtype=float) for r in rows])

                entry = {
                    "name": f"{pid}_{tr}_{inst}",
                    "participant": pid,
                    "trial": tr,
                    "instrument": inst,
                    "expertise": exp_str,
                    "level": lvl_str,
                    "expertise_idx": y_exp,
                    "level_idx": y_lvl,
                    "data": M.tolist()
                }
                entries.append(entry); kept += 1
                continue
            # ---------- MONO ----------
            Ts   = _first(B, "timestamp")
            Pos  = _first(B, "position")

            # Flags: prefer captured_flag, then inuse_flag, then tracking_flag
            Cap = _first(B, "captured_flag")
            if Cap is None:
                Cap = _first(B, "inuse_flag")
            if Cap is None:
                Cap = _first(B, "tracking_flag")

            # Position is required for mono entries
            if Pos is None:
                skipped += 1
                skip_reasons["mono_missing_position"] += 1
                continue

            # Parse arrays (Ts is optional)
            try:
                P  = _as_pos_3xN(Pos)              # 3 x Np
                Np = P.shape[1]
                Cap_arr = _as_1d(Cap, bool) if Cap is not None else None
                Ts_arr  = _as_1d(Ts,  float) if Ts  is not None else None
            except Exception:
                skipped += 1
                skip_reasons["mono_bad_shapes"] += 1
                continue

            # Détermine la longueur finale N à partir de ce qui est dispo
            candidates = [Np]
            if Cap_arr is not None:
                candidates.append(len(Cap_arr))
            if Ts_arr is not None:
                candidates.append(len(Ts_arr))
            N = int(min(candidates)) if candidates else 0
            if N == 0:
                skipped += 1
                skip_reasons["mono_zero_length"] += 1
                continue

            # Coupe / pad aux mêmes longueurs
            Pn3 = P.T[:N]                 # (N,3)
            if Cap_arr is None:
                Cap_arr = np.ones(N, dtype=bool)
            else:
                Cap_arr = Cap_arr[:N].astype(bool)

            if Ts_arr is None:
                Ts_arr = np.arange(N, dtype=float) / SAMPLE_RATE_HZ
            else:
                Ts_arr = Ts_arr[:N].astype(float)

            # X, Y, Z + magnitude
            P = Pn3.T                     # (3,N)
            Xp, Yp, Zp = P[0], P[1], P[2]
            pos_mag = np.sqrt(Xp*Xp + Yp*Yp + Zp*Zp)

            # Masquage strict par Cap_arr
            Xp = np.where(Cap_arr, Xp, 0.0)
            Yp = np.where(Cap_arr, Yp, 0.0)
            Zp = np.where(Cap_arr, Zp, 0.0)
            pos_mag = np.where(Cap_arr, pos_mag, 0.0)

            # Metrics mono: si absent -> zeros (pas de dérivation)
            def take_or_zeros(key):
                arr = _first(B, key)
                if arr is None:
                    return np.zeros(N, dtype=float)
                arr = _as_1d(arr, float)
                if len(arr) >= N:
                    return arr[:N]
                out = np.zeros(N, dtype=float)
                out[:len(arr)] = arr
                return out

            vel  = take_or_zeros("velocity")
            acc  = take_or_zeros("acceleration")
            jerk = take_or_zeros("jerk")

            # Distances injectées si présentes côté bimanual, masquées par leurs propres flags
            _tmp = read_dist("bipolar_to_cavitron", N)
            dist_bip_cav = _tmp if _tmp is not None else np.zeros(N, dtype=float)
            _tmp = read_dist("bipolar_to_scissors", N)
            dist_bip_sci = _tmp if _tmp is not None else np.zeros(N, dtype=float)

            rows = [Xp, Yp, Zp, pos_mag, vel, acc, jerk, dist_bip_cav, dist_bip_sci]
            label_row_exp = np.full(N, y_exp, dtype=float)
            label_row_lvl = np.full(N, y_lvl, dtype=float)
            M = np.vstack([label_row_exp, label_row_lvl] + [np.asarray(r, dtype=float) for r in rows])

            entry = {
                "name": f"{pid}_{tr}_{inst}",
                "participant": pid,
                "trial": tr,
                "instrument": inst,
                "expertise": exp_str,
                "level": lvl_str,
                "expertise_idx": y_exp,
                "level_idx": y_lvl,
                "data": M.tolist()
            }
            entries.append(entry); kept += 1

    print(f"✅ Built entries: {kept}  |  Skipped: {skipped}")
    if skip_reasons:
        print("↪️  Skip breakdown:", dict(skip_reasons))

    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(entries, f)
    print(f"[OK] PKL → {out_pkl}")

    metric_names = [
        "Label(Expertise)",
        "Label(Level)",
        "Position Magnitude",
        "Velocity",
        "Acceleration",
        "Jerk",
        "Distance Bipolar–Cavitron",
        "Distance Bipolar–Scissors",
    ]
    meta = {
        "metric_names": metric_names,
        "label_map_expertise": label_map_expertise,
        "label_map_level": label_map_level,
        "policy": {
            "mask_single": "captured_flag False => zeros",
            "mask_distance": "bidist_captured_flag (fallback captured_flag) False => zeros",
            "no_derivation": "Si velocity/acc/jerk absents => zeros",
            "timestamps_bimanual": "repris depuis bipolar; sinon synthétiques à 10 Hz"
        },
        "sample_rate_hz": SAMPLE_RATE_HZ
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] META → {out_meta}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_json", type=str, default="data/full_data.json")
    ap.add_argument("--out_pkl",   type=str, default=os.path.join(".", "data", "final_from_full_B.pkl"))
    args = ap.parse_args()

    out_meta = os.path.splitext(args.out_pkl)[0] + "_meta.json"
    build_dataset(args.full_json, args.out_pkl, out_meta)

if __name__ == "__main__":
    main()
