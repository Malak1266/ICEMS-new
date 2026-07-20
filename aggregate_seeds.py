"""
aggregate_seeds.py — Reduction de dispersion SANS fuite + figure bornee + rapport d'ordinalite.

PRINCIPE DE NON-FUITE (a lire avant d'utiliser)
-----------------------------------------------
Aucune operation de ce script ne consulte les labels Junior / Senior.
  * Le moyennage multi-seeds ne depend d'aucun label.
  * La calibration est ajustee UNIQUEMENT sur les participants Novice/Expert
    du SPLIT TRAIN (colonne `split`), puis appliquee telle quelle a tout le monde.
  * Junior/Senior ne servent QU'A l'evaluation finale, jamais au reglage.
=> Junior et Senior restent des predictions libres.

ENTREE ATTENDUE
---------------
Un CSV par seed, ex. runs/seed_0.csv, runs/seed_1.csv, ... contenant :
    participant , group , split , raw_score
  - group in {Novice, Junior, Senior, Expert}
  - split in {train, val, test}   (pour Junior/Senior mettre 'test')
  - raw_score = score composite BRUT dans (-1,1)  [moyenne 10 paires x 3 essais]

USAGE
-----
python aggregate_seeds.py --glob "runs/seed_*.csv" --out panelA_final
python aggregate_seeds.py --glob "runs/seed_*.csv" --out panelA_raw --calib raw
"""
from __future__ import annotations
import argparse, glob, sys
import numpy as np
import pandas as pd
from calibration import fit_bounded_affine, apply_bounded_affine

GROUPS = ["Novice", "Junior", "Senior", "Expert"]


def load_seeds(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"Aucun fichier ne correspond a : {pattern}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        need = {"participant", "group", "raw_score"}
        if not need.issubset(d.columns):
            sys.exit(f"{f} : colonnes requises {need}, trouvees {set(d.columns)}")
        if "split" not in d.columns:
            print(f"[ATTENTION] {f} sans colonne `split` -> calibration ajustee sur "
                  f"TOUS les extremes (moins rigoureux, a declarer).")
            d["split"] = "unknown"
        d["seed_file"] = f
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f"{len(files)} seeds chargees, {df.participant.nunique()} participants.")
    return df


def average_seeds(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["participant", "group", "split"], as_index=False)
           .agg(raw_score=("raw_score", "mean"),
                sd_seeds=("raw_score", "std"),
                n_seeds=("raw_score", "size")))
    g["sd_seeds"] = g["sd_seeds"].fillna(0.0)
    print(f"Ecart-type inter-seeds moyen : {g.sd_seeds.mean():.4f} "
          f"(bruit d'optimisation elimine par le moyennage)")
    return g


def calibrate_train_only(df: pd.DataFrame, mode: str = "bounded",
                         anchor_lo: float = -0.9, anchor_hi: float = 0.9) -> pd.DataFrame:
    df = df.copy()
    if mode == "raw":
        df["score"] = df["raw_score"]
        print("Calibration : AUCUNE (scores tanh bruts). Protocole le plus proche du papier.")
        return df
    fit_mask = df["group"].isin(["Novice", "Expert"])
    if (df["split"] == "train").any():
        fit_mask = fit_mask & (df["split"] == "train")
        scope = "extremes du SPLIT TRAIN uniquement (pas de fuite)"
    else:
        scope = "tous les extremes (colonne `split` absente)"
    fit = df[fit_mask]
    n_nov = int((fit.group == "Novice").sum())
    n_exp = int((fit.group == "Expert").sum())
    if n_nov < 2 or n_exp < 2:
        sys.exit(f"Ajustement impossible : {n_nov} Novices / {n_exp} Experts dans le perimetre.")
    a, b = fit_bounded_affine(fit["raw_score"].values, fit["group"].values,
                              anchor_lo=anchor_lo, anchor_hi=anchor_hi)
    df["score"] = apply_bounded_affine(df["raw_score"].values, a, b)
    print(f"Calibration bornee atanh->tanh : a={a:.4f}, b={b:.4f}")
    print(f"  ajustee sur {scope} ({n_nov} Novices, {n_exp} Experts)")
    assert df.score.between(-1, 1).all(), "ECHEC bornage"
    return df


def report(df: pd.DataFrame, path: str, calib: str) -> str:
    from scipy import stats
    L = []
    A = L.append
    A("RAPPORT — dispersion, ordinalite, contrastes")
    A("=" * 62)
    A(f"calibration : {calib}")
    A(f"seeds moyennees : {int(df.n_seeds.max())}")
    A(f"ecart-type inter-seeds moyen : {df.sd_seeds.mean():.4f}")
    A("")
    A("Par groupe (le nerf de la guerre = SD intra-classe)")
    A(f"{'groupe':<10}{'n':>4}{'moyenne':>10}{'SD':>9}{'IC95 bas':>11}{'IC95 haut':>11}")
    stats_g = {}
    for g in GROUPS:
        v = df.loc[df.group == g, "score"].values
        if len(v) == 0:
            continue
        m = v.mean()
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        half = 1.96 * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
        stats_g[g] = (m, sd, len(v))
        A(f"{g:<10}{len(v):>4}{m:>10.3f}{sd:>9.3f}{m-half:>11.3f}{m+half:>11.3f}")
    A("")
    A("Ordinalite attendue : Novice < Junior < Senior < Expert")
    ok_all = True
    for lo, hi in zip(GROUPS[:-1], GROUPS[1:]):
        if lo in stats_g and hi in stats_g:
            ok = stats_g[lo][0] < stats_g[hi][0]
            ok_all = ok_all and ok
            A(f"  {lo:<7} < {hi:<7} : {'OUI' if ok else 'NON'}   "
              f"({stats_g[lo][0]:+.3f} vs {stats_g[hi][0]:+.3f})")
    A(f"  => ordre complet respecte : {'OUI' if ok_all else 'NON'}")
    A("")
    A("Contrastes (Hedges' g + Welch). Novice/Expert ancres => tautologiques.")
    A(f"{'contraste':<20}{'g':>8}{'p':>12}   lecture")
    for lo, hi in [("Novice", "Junior"), ("Junior", "Senior"), ("Senior", "Expert"),
                   ("Novice", "Expert"), ("Novice", "Senior"), ("Junior", "Expert")]:
        if lo not in stats_g or hi not in stats_g:
            continue
        x = df.loc[df.group == lo, "score"].values
        y = df.loc[df.group == hi, "score"].values
        if len(x) < 2 or len(y) < 2:
            continue
        n1, n2 = len(x), len(y)
        sp = np.sqrt(((n1-1)*x.var(ddof=1) + (n2-1)*y.var(ddof=1)) / (n1+n2-2))
        d = (y.mean() - x.mean()) / sp if sp > 0 else np.nan
        J = 1 - 3.0/(4*(n1+n2) - 9)
        p = stats.ttest_ind(x, y, equal_var=False).pvalue
        if {lo, hi} == {"Novice", "Expert"}:
            tag = "TAUTOLOGIQUE (ancre)"
        elif {lo, hi} <= {"Junior", "Senior"}:
            tag = "libre"
        else:
            tag = "semi-libre"
        A(f"{lo+'-'+hi:<20}{d*J:>8.2f}{p:>12.2e}   {tag}")
    A("")
    A("Rappel : aucun reglage n'a consulte les labels Junior/Senior.")
    txt = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return txt


def figures(df: pd.DataFrame, out: str, calib: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    try:
        from palette_icems import PALETTE, INK, apply_style
        apply_style()
    except ImportError:
        PALETTE = {"Novice": "#c9d3d9", "Junior": "#9fb4bf",
                   "Senior": "#5f7d8c", "Expert": "#2f4a58"}
        INK = "#2b2b2b"
        rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    present = [g for g in GROUPS if (df.group == g).any()]
    data = [df.loc[df.group == g, "score"].values for g in present]
    nseed = int(df.n_seeds.max())

    means, half = [], []
    for v in data:
        means.append(v.mean())
        half.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    x = np.arange(len(present))
    ax.bar(x, means, color=[PALETTE[g] for g in present], edgecolor=INK, width=0.62, zorder=3)
    ax.errorbar(x, means, yerr=np.array([half, half]), fmt="none", ecolor=INK,
                elinewidth=1.3, capsize=5, zorder=4)
    ax.axhline(0, color="#9aa7ad", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(present)
    ax.set_ylim(-1, 1); ax.set_yticks(np.arange(-1, 1.01, 0.5))
    ax.set_ylabel("Score d'expertise")
    fig.subplots_adjust(top=0.84, left=0.16, right=0.96, bottom=0.11)
    fig.suptitle("Score moyen par niveau — Hybrid 1", y=0.965, fontsize=12.5)
    fig.text(0.5, 0.90, f"moyenne de {nseed} seeds · calibration = {calib}",
             ha="center", fontsize=9.5, color="#666", style="italic")
    fig.savefig(f"{out}_barres.pdf"); fig.savefig(f"{out}_barres.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.9, 5.4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.62,
                    medianprops=dict(color=INK, lw=1.6),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor="white", markeredgecolor=INK))
    for patch, g in zip(bp["boxes"], present):
        patch.set_facecolor(PALETTE[g]); patch.set_edgecolor(INK)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, 1):
        ax.scatter(np.full(len(v), i) + rng.normal(0, 0.05, len(v)), v,
                   s=18, facecolor="white", edgecolor=INK, linewidth=0.6, zorder=5)
    ax.axhline(0, color="#c3ccd1", lw=0.8)
    ax.set_xticks(range(1, len(present)+1)); ax.set_xticklabels(present)
    ax.set_ylim(-1, 1); ax.set_yticks(np.arange(-1, 1.01, 0.5))
    ax.set_ylabel("Score d'expertise")
    fig.subplots_adjust(top=0.84, left=0.16, right=0.96, bottom=0.11)
    fig.suptitle("Distribution des scores par niveau — Hybrid 1", y=0.965, fontsize=12.5)
    fig.text(0.5, 0.90, f"moyenne de {nseed} seeds · calibration = {calib}",
             ha="center", fontsize=9.5, color="#666", style="italic")
    fig.savefig(f"{out}_boxplot.pdf"); fig.savefig(f"{out}_boxplot.png", dpi=300)
    plt.close(fig)
    print(f"figures : {out}_barres.pdf/.png  et  {out}_boxplot.pdf/.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help='ex. "runs/seed_*.csv"')
    p.add_argument("--out", default="panelA_final")
    p.add_argument("--calib", default="bounded", choices=["raw", "bounded"])
    p.add_argument("--anchor-lo", type=float, default=-0.9)
    p.add_argument("--anchor-hi", type=float, default=0.9)
    a = p.parse_args()
    df = average_seeds(load_seeds(a.glob))
    df = calibrate_train_only(df, mode=a.calib,
                              anchor_lo=a.anchor_lo, anchor_hi=a.anchor_hi)
    assert df.score.between(-1, 1).all(), "depassement des bornes"
    df.to_csv(f"{a.out}_scores.csv", index=False)
    print("\n" + report(df, f"{a.out}_rapport.txt", a.calib))
    figures(df, a.out, a.calib)


if __name__ == "__main__":
    main()
