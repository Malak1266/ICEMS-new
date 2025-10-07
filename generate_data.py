# -*- coding: utf-8 -*-
"""
Generate dataset (Format A: list[dict]) from full_data.json for ICEMS.

What it does (aligned with user's spec):
- Includes ALL metrics present in the original streams that we need:
  * X, Y, Z Position (from "position")
  * Position Magnitude (derived)
  * Velocity, Acceleration, Jerk (read if present; otherwise computed from position)
  * Distance Bipolar–Cavitron, Distance Bipolar–Scissors (when available)
- Uses ONLY the captured flags to filter/mask the data:
  * For single-instrument metrics (pos/vel/acc/jerk): captured_flag
  * For bimanual metrics (distances): bidist_captured_flag (fallback to captured_flag)
- Preserves time alignment (10 Hz) and keeps constant length series by zeroing values where flags=False.
  (We then drop windows with >=50% zeros later during training, as requested.)
- Adds two label rows: Label(Expertise) and Label(Level) (per time-step, constant across a trial).

Outputs:
- PKL file: list of dict entries:
    {
      "name": "<participant>_<trial>_<instrument>",
      "participant": "<id>",
      "trial": "<trial>",
      "instrument": "<instrument>",
      "expertise": "<str>",
      "level": "<str>",
      "expertise_idx": <int>,
      "level_idx": <int>,
      "data": [[... rows ...] x T]
    }
- META json file with metric_names (row order) and label maps.

Usage:
    python generate_data.py \
        --full_json /path/to/full_data.json \
        --out_pkl   /path/to/final_from_full_A.pkl

Defaults are reasonable for running inside this repo.
"""

import os
import json
import pickle
import argparse
import numpy as np
from collections import defaultdict

SAMPLE_RATE_HZ = 10.0
DEFAULT_DT = 1.0 / SAMPLE_RATE_HZ

# --------------------- small utils ---------------------
def _as_bool(x):
    a = np.asarray(x)
    return a.astype(bool) if a.dtype != bool else a

def _as_1d(x, dtype=float):
    a = np.asarray(x, dtype=dtype)
    if a.ndim != 1:
        raise ValueError(f"expected 1D, got shape={a.shape}")
    return a

def _as_3xn(mat):
    A = np.asarray(mat, dtype=float)
    if A.ndim != 2:
        raise ValueError("position must be 2D")
    # accept (N,3) or (3,N)
    if A.shape[1] == 3:   # (N,3)
        return A.T
    if A.shape[0] == 3:   # (3,N)
        return A
    raise ValueError(f"position not 3D compatible: {A.shape}")

def _align_min_len(arrs):
    lens = [len(x) for x in arrs if x is not None and len(x) > 0]
    if not lens:
        return [], 0
    N = int(min(lens))
    out = []
    for x in arrs:
        if x is None or len(x) == 0:
            out.append(None)
        else:
            out.append(x[:N])
    return out, N

def _get_dt(t):
    if t is None or len(t) < 2:
        return DEFAULT_DT
    d = np.diff(np.asarray(t, dtype=float))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.mean(d)) if len(d) else DEFAULT_DT

def _take_first_or_none(lst):
    if lst is None:
        return None
    if isinstance(lst, list) and len(lst) > 0:
        return lst[0]
    return lst

def _get_bundle_field(bundle, name):
    arrs = bundle.get(name, None)
    if arrs is None:
        return None
    return _take_first_or_none(arrs)

def _fit_to_length(x, L):
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    if len(x) >= L:
        return x[:L]
    out = np.zeros(L, dtype=float)
    out[:len(x)] = x
    return out

def _fit_mask_to_length(m, L):
    if m is None:
        return None
    m = np.asarray(m, dtype=bool)
    if len(m) >= L:
        return m[:L]
    out = np.zeros(L, dtype=bool)
    out[:len(m)] = m[:len(m)]
    return out

def _mag3(M3xN):
    return np.sqrt(np.sum(M3xN[:3]**2, axis=0))

def _central_diff(x, dt):
    # central differences with forward/backward at ends
    x = np.asarray(x, dtype=float)
    v = np.zeros_like(x)
    if len(x) >= 3:
        v[1:-1] = (x[2:] - x[:-2]) / (2.0*dt)
        v[0]    = (x[1] - x[0]) / dt
        v[-1]   = (x[-1] - x[-2]) / dt
    elif len(x) == 2:
        v[0]  = (x[1]-x[0])/dt
        v[1]  = v[0]
    return v

def _derive_speed_from_pos(P3xN, dt):
    # 3D velocity magnitude, acceleration magnitude, jerk magnitude
    x,y,z = P3xN[0], P3xN[1], P3xN[2]
    vx = _central_diff(x, dt); vy = _central_diff(y, dt); vz = _central_diff(z, dt)
    vmag = np.sqrt(vx*vx + vy*vy + vz*vz)
    ax = _central_diff(vx, dt); ay = _central_diff(vy, dt); az = _central_diff(vz, dt)
    amag = np.sqrt(ax*ax + ay*ay + az*az)
    jx = _central_diff(ax, dt); jy = _central_diff(ay, dt); jz = _central_diff(az, dt)
    jmag = np.sqrt(jx*jx + jy*jy + jz*jz)
    return vmag, amag, jmag

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
    by_combo = defaultdict(dict)  # (pid,trial,instrument) -> metrics dict name->[arr,...]
    labels_by_combo = {}

    for i in ids:
        pid  = D["participant"][i]
        tr   = D["trial"][i]
        inst = D["instrument"][i]
        met  = D["metric"][i]
        arr  = D["data"][i]
        key  = (str(pid), str(tr), str(inst))

        labels_by_combo[key] = ((D["expertise"][i] or "").strip(),
                                (D["level"][i] or "").strip())

        by_combo[key].setdefault(met, []).append(arr)

    # prepare label maps
    all_exp = sorted({labels_by_combo[k][0] for k in labels_by_combo})
    all_lvl = sorted({labels_by_combo[k][1] for k in labels_by_combo})
    label_map_expertise = {e:i for i,e in enumerate(all_exp)}
    label_map_level     = {e:i for i,e in enumerate(all_lvl)}
    print("🧭 expertise map:", label_map_expertise)
    print("🧭 level map    :", label_map_level)

    # index by (pid,trial) across instruments
    by_pt = defaultdict(dict)
    for (pid,tr,inst), bundle in by_combo.items():
        by_pt[(pid,tr)][inst] = bundle

    entries = []
    kept = skipped = 0

    for (pid,tr), inst_map in by_pt.items():
        for inst, B in inst_map.items():
            exp_str, lvl_str = labels_by_combo[(pid,tr,inst)]
            y_exp = label_map_expertise.get(exp_str, 0)
            y_lvl = label_map_level.get(lvl_str, 0)

            Ts   = _get_bundle_field(B, "timestamp")
            Pos  = _get_bundle_field(B, "position")
            Cap  = _get_bundle_field(B, "captured_flag")    # mono-mask
            if Ts is None or Pos is None or Cap is None:
                skipped += 1
                continue

            try:
                Ts  = _as_1d(Ts, dtype=float)
                P3  = _as_3xn(Pos)      # (3,N)
                Cap = _as_bool(Cap)     # (N,)
            except Exception:
                skipped += 1
                continue

            # align to common length for mono-instrument channels
            arrs, Nmono = _align_min_len([Ts, P3.T, Cap])
            if Nmono == 0:
                skipped += 1; continue
            Ts   = arrs[0]
            Pn3  = arrs[1]            # (N,3) aligned
            Cap0 = _as_bool(arrs[2])  # (N,)

            P3 = Pn3.T                # back to (3,N)
            dt = _get_dt(Ts)

            # X,Y,Z (masked by captured_flag -> zeros outside capture)
            Xp = P3[0]; Yp = P3[1]; Zp = P3[2]
            pos_mag = _mag3(P3)
            Xp = np.where(Cap0, Xp[:Nmono], 0.0)
            Yp = np.where(Cap0, Yp[:Nmono], 0.0)
            Zp = np.where(Cap0, Zp[:Nmono], 0.0)
            pos_mag = np.where(Cap0, pos_mag[:Nmono], 0.0)

            # Velocity/Acceleration/Jerk: prefer streams if present, else derive
            vel_stream  = _get_bundle_field(B, "velocity")
            acc_stream  = _get_bundle_field(B, "acceleration")
            jerk_stream = _get_bundle_field(B, "jerk")

            if vel_stream is not None and acc_stream is not None and jerk_stream is not None:
                vel  = _fit_to_length(_as_1d(vel_stream, dtype=float), Nmono)
                acc  = _fit_to_length(_as_1d(acc_stream, dtype=float), Nmono)
                jerk = _fit_to_length(_as_1d(jerk_stream, dtype=float), Nmono)
            else:
                vmag, amag, jmag = _derive_speed_from_pos(P3, dt)
                vel  = _fit_to_length(vmag, Nmono)
                acc  = _fit_to_length(amag, Nmono)
                jerk = _fit_to_length(jmag, Nmono)

            vel  = np.where(Cap0, vel, 0.0)
            acc  = np.where(Cap0, acc, 0.0)
            jerk = np.where(Cap0, jerk, 0.0)

            # Bimanual distances (choose best available instrument bundle and mask with bidist flag)
            def _read_dist(inst_name):
                b = inst_map.get(inst_name, None)
                if b is None: return None, None
                dist = _get_bundle_field(b, "distance")
                if dist is None: return None, None
                bid  = _get_bundle_field(b, "bidist_captured_flag")
                capb = _get_bundle_field(b, "captured_flag")
                mask = bid if bid is not None else capb
                if mask is None: return None, None
                dist = _as_1d(dist, dtype=float)
                mask = _as_bool(mask)
                aligned, NN = _align_min_len([dist, mask])
                if NN == 0: return None, None
                dist_al = aligned[0]; m_al = _as_bool(aligned[1])
                return dist_al, m_al

            def _pair(a,b):
                da,ma = _read_dist(a)
                db,mb = _read_dist(b)
                na = int(ma.sum()) if ma is not None else 0
                nb = int(mb.sum()) if mb is not None else 0
                if na >= nb and da is not None: return da,ma
                if db is not None: return db,mb
                return None,None

            dist_bip_cav, mask_bc = _pair("bipolar","cavitron")
            dist_bip_sci, mask_bs = _pair("bipolar","scissors")

            # align distances to Nmono and zero outside their own masks
            if dist_bip_cav is not None:
                dist_bip_cav = _fit_to_length(dist_bip_cav, Nmono)
                mask_bc = np.ones_like(dist_bip_cav, bool) if mask_bc is None else _fit_mask_to_length(mask_bc, Nmono)
                dist_bip_cav = np.where(mask_bc, dist_bip_cav, 0.0)
            else:
                dist_bip_cav = np.zeros(Nmono, dtype=float)

            if dist_bip_sci is not None:
                dist_bip_sci = _fit_to_length(dist_bip_sci, Nmono)
                mask_bs = np.ones_like(dist_bip_sci, bool) if mask_bs is None else _fit_mask_to_length(mask_bs, Nmono)
                dist_bip_sci = np.where(mask_bs, dist_bip_sci, 0.0)
            else:
                dist_bip_sci = np.zeros(Nmono, dtype=float)

            # Assemble rows in final order
            rows = [
                Xp, Yp, Zp,            # 0..2
                pos_mag,               # 3
                vel, acc, jerk,        # 4..6
                dist_bip_cav,          # 7
                dist_bip_sci,          # 8
            ]

            T = Nmono
            label_row_exp = np.full(T, int(y_exp), dtype=float)
            label_row_lvl = np.full(T, int(y_lvl), dtype=float)
            M = np.vstack([label_row_exp, label_row_lvl] + [np.asarray(r, dtype=float) for r in rows])

            entry = {
                "name": f"{pid}_{tr}_{inst}",
                "participant": pid,
                "trial": tr,
                "instrument": inst,
                "expertise": exp_str,
                "level": lvl_str,
                "expertise_idx": int(y_exp),
                "level_idx": int(y_lvl),
                "data": M.tolist()
            }
            entries.append(entry); kept += 1

    print(f"✅ Built entries: {kept}  |  Skipped (missing critical streams): {skipped}")

    # save
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(entries, f)
    print(f"[OK] PKL → {out_pkl}")

    # meta
    metric_names = [
        "Label(Expertise)",
        "Label(Level)",
        "X Position",
        "Y Position",
        "Z Position",
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
        "flags_policy": {
            "single_instrument": "captured_flag used to mask (zeros when False)",
            "bimanual_distance": "bidist_captured_flag (fallback captured_flag) used to mask (zeros when False)",
            "note": "We do NOT use 'inuse' or 'tracking' for filtering, per spec."
        },
        "sample_rate_hz": SAMPLE_RATE_HZ
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] META → {out_meta}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_json", type=str, default="data/full_data.json")
    ap.add_argument("--out_pkl",   type=str, default=os.path.join(".", "data", "final_from_full_A.pkl"))
    args = ap.parse_args()

    out_meta = os.path.splitext(args.out_pkl)[0] + "_meta.json"
    build_dataset(args.full_json, args.out_pkl, out_meta)

if __name__ == "__main__":
    main()
