#!/usr/bin/env python3
"""
audit_pipeline.py — Audit de provenance, de fuite et de reproductibilite du pipeline ICEMS.

A EXECUTER DANS LA RACINE DU DEPOT (la ou se trouve src/ et experiments/).
Ne modifie AUCUN fichier du projet. Ecrit un seul rapport : docs/PROVENANCE.md

    python audit_pipeline.py --repo . --out docs/PROVENANCE.md

Ce que l'auditeur etablit
-------------------------
  [A] Inventaire signe   : MD5 de chaque tenseur, checkpoint, CSV de scores
  [B] Version tenseur    : qui utilise v1, qui utilise v2 (le confond captured_flag)
  [C] Integrite du split : chevauchement de participants entre train/val/test  -> FUITE
  [D] Etancheite middle  : un Junior/Senior apparait-il en train ?             -> FUITE GRAVE
  [E] Normalisation      : statistiques calculees train-only ou sur tout ?     -> FUITE SUBTILE
  [F] Calibration        : ajustee sur le train seul ou sur tous les extremes ?
  [G] Convention temps   : sens de l'axe, niveau d'agregation (essai vs participant)
  [H] Reproductibilite   : seeds fixees, versions de librairies, etat git

Chaque verification renvoie PASS / FAIL / INCONNU. "INCONNU" n'est pas "PASS".
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, datetime
from pathlib import Path

TENSOR_PAT = re.compile(r"trial_tensor_v(\d)\.pkl")
GROUPS_MIDDLE = {"junior", "senior"}
GROUPS_EXTREME = {"novice", "expert"}


# ----------------------------------------------------------------- utilitaires
def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception as e:
        return f"ERREUR:{e}"


def mtime(path: Path) -> str:
    try:
        return datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return "?"


def size_mb(path: Path) -> str:
    try:
        return f"{path.stat().st_size/1e6:.2f} Mo"
    except Exception:
        return "?"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


class Report:
    def __init__(self):
        self.lines = []
        self.verdicts = []

    def h(self, t, lvl=2):
        self.lines.append(f"\n{'#'*lvl} {t}\n")

    def p(self, t=""):
        self.lines.append(t)

    def check(self, code, label, status, detail=""):
        icon = {"PASS": "OK", "FAIL": "ECHEC", "WARN": "ALERTE", "INCONNU": "INCONNU"}[status]
        self.verdicts.append((code, label, status))
        self.p(f"- **[{code}] {label}** : `{icon}`" + (f" — {detail}" if detail else ""))

    def dump(self, path: Path):
        head = ["# Rapport de provenance et d'audit — ICEMS",
                f"\nGenere le {datetime.datetime.now().isoformat(timespec='seconds')}",
                f"\nMachine : `{os.uname().nodename if hasattr(os,'uname') else 'windows'}`",
                f"\nPython : `{sys.version.split()[0]}`\n"]
        n_fail = sum(1 for _, _, s in self.verdicts if s == "FAIL")
        n_unk = sum(1 for _, _, s in self.verdicts if s == "INCONNU")
        head.append(f"\n**Synthese : {n_fail} ECHEC · {n_unk} INCONNU · "
                    f"{len(self.verdicts)} verifications**\n")
        if n_fail == 0 and n_unk == 0:
            head.append("\n> Pipeline verrouille. Le run peut etre lance.\n")
        else:
            head.append("\n> **NE PAS LANCER LE RUN** tant qu'il reste un ECHEC ou un INCONNU.\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(head + self.lines), encoding="utf-8")


# ------------------------------------------------------------- [A] inventaire
def inventory(repo: Path, R: Report):
    R.h("[A] Inventaire signe des artefacts")
    pats = ["**/*.pkl", "**/*.pt", "**/*.h5", "**/*.weights.h5", "**/*.keras", "**/*.ckpt"]
    found = []
    for pat in pats:
        for f in repo.glob(pat):
            if any(x in f.parts for x in (".git", "node_modules", ".venv")):
                continue
            found.append(f)
    if not found:
        R.p("_Aucun tenseur ni checkpoint trouve sous la racine indiquee._")
        R.check("A1", "Artefacts presents", "INCONNU", "verifier --repo")
        return {}
    R.p("| fichier | taille | modifie le | MD5 |")
    R.p("|---|---|---|---|")
    table = {}
    for f in sorted(found)[:80]:
        h = md5(f)
        table[str(f.relative_to(repo))] = h
        R.p(f"| `{f.relative_to(repo)}` | {size_mb(f)} | {mtime(f)} | `{h}` |")
    R.check("A1", "Artefacts presents", "PASS", f"{len(found)} fichiers signes")

    # doublons = meme contenu sous deux noms (checkpoint recopie)
    inv = {}
    for k, v in table.items():
        inv.setdefault(v, []).append(k)
    dups = {v: ks for v, ks in inv.items() if len(ks) > 1}
    if dups:
        R.p("\n**Fichiers identiques (meme MD5) :**")
        for v, ks in dups.items():
            R.p(f"- `{v[:12]}…` : " + ", ".join(f"`{k}`" for k in ks))
        R.check("A2", "Doublons de checkpoint", "WARN",
                "deux noms differents pointent le meme contenu")
    else:
        R.check("A2", "Doublons de checkpoint", "PASS")
    return table


# --------------------------------------------------------- [B] version tenseur
def tensor_version(repo: Path, R: Report):
    R.h("[B] Version du tenseur utilisee (confond captured_flag v1)")
    hits = {"1": [], "2": []}
    continuous_hits = []
    for f in repo.rglob("*.py"):
        if any(x in f.parts for x in (".git", ".venv", "node_modules")) or f.name == "audit_pipeline.py":
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in TENSOR_PAT.finditer(txt):
            hits[m.group(1)].append(str(f.relative_to(repo)))
        if "continuous_per_trial.pkl" in txt:
            continuous_hits.append(str(f.relative_to(repo)))
    for v in ("1", "2"):
        if hits[v]:
            R.p(f"\n**v{v}** reference dans :")
            for f in sorted(set(hits[v])):
                R.p(f"- `{f}`")
    if continuous_hits:
        R.p("\n**continuous_per_trial.pkl** (pipeline extremes Option A) reference dans :")
        for f in sorted(set(continuous_hits))[:20]:
            R.p(f"- `{f}`")
    if hits["1"] and hits["2"]:
        R.check("B1", "Version tenseur unique", "FAIL",
                "v1 ET v2 coexistent — les figures ne sont pas comparables")
    elif hits["1"] and not continuous_hits:
        R.check("B1", "Version tenseur unique", "FAIL",
                "v1 seul : confond activite/detection non corrige")
    elif hits["2"] and not hits["1"]:
        R.check("B1", "Version tenseur unique", "PASS", "v2 partout")
    elif continuous_hits and not hits["1"]:
        R.check("B1", "Version tenseur unique", "PASS",
                "continuous_per_trial.pkl (pipeline extremes)")
    elif hits["1"] and continuous_hits:
        R.check("B1", "Version tenseur unique", "FAIL",
                "v1 et continuous_per_trial coexistent")
    else:
        R.check("B1", "Version tenseur unique", "INCONNU", "aucune reference trouvee")


# ------------------------------------------------- [C][D] integrite des splits
def split_integrity(repo: Path, R: Report, glob_pat: str):
    R.h("[C] Integrite du split — chevauchement de participants")
    try:
        import pandas as pd
    except ImportError:
        R.check("C1", "Chevauchement train/test", "INCONNU", "pandas absent")
        return
    files = sorted(repo.glob(glob_pat))
    if not files:
        R.check("C1", "Chevauchement train/test", "INCONNU",
                f"aucun CSV ne correspond a `{glob_pat}`")
        R.check("D1", "Etancheite Junior/Senior", "INCONNU", "idem")
        return
    ok_c = ok_d = True
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as e:
            R.p(f"- `{f.name}` illisible : {e}")
            continue
        if not {"participant", "group", "split"}.issubset(d.columns):
            R.p(f"- `{f.name}` : colonnes manquantes {set(d.columns)}")
            ok_c = ok_d = False
            continue
        d["_g"] = d.group.astype(str).str.strip().str.lower()
        sets = {s: set(d.loc[d.split == s, "participant"]) for s in ("train", "val", "test")}
        inter_tt = sets["train"] & sets["test"]
        inter_tv = sets["train"] & sets["val"]
        if inter_tt or inter_tv:
            ok_c = False
            R.p(f"- `{f.name}` : train∩test={sorted(inter_tt)} train∩val={sorted(inter_tv)}")
        # [D] un middle en train ?
        mid_in_train = set(d.loc[(d.split == "train") & d._g.isin(GROUPS_MIDDLE), "participant"])
        if mid_in_train:
            ok_d = False
            R.p(f"- `{f.name}` : **Junior/Senior en TRAIN** -> {sorted(mid_in_train)}")
    R.check("C1", "Aucun participant partage entre splits",
            "PASS" if ok_c else "FAIL",
            "pseudo-replication si ECHEC" if not ok_c else "")
    R.h("[D] Etancheite du pool intermediaire")
    R.check("D1", "Aucun Junior/Senior dans le train",
            "PASS" if ok_d else "FAIL",
            "fuite grave : les predictions middle ne sont plus libres" if not ok_d else "")


# ------------------------------------ [E][F] normalisation et calibration
# Fichiers critiques du pipeline extremes (Option A) — F1 ne bloque que sur ceux-la
_CALIB_CRITICAL = {
    "calibration.py",
    "aggregate_seeds.py",
    "experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/train.py",
}


def _skip_py(f: Path) -> bool:
    return any(x in f.parts for x in (".git", ".venv", "node_modules")) or f.name == "audit_pipeline.py"


def norm_and_calib(repo: Path, R: Report):
    R.h("[E] Normalisation — statistiques train-only ?")
    pat_all = re.compile(r"(StandardScaler\(\)\s*\.fit_transform\(\s*X\b|"
                         r"\.fit\(\s*X\s*\)|mean\(axis=\(0,\s*1\)\))")
    pat_train = re.compile(r"(fit\(\s*X_?tr|compute_train_norm_stats|fit\(\s*X_train)|"
                           r"compute_norm_stats\(")
    suspicious, good = [], []
    for f in repo.rglob("*.py"):
        if _skip_py(f):
            continue
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pat_train.search(t):
            good.append(str(f.relative_to(repo)))
        elif pat_all.search(t):
            suspicious.append(str(f.relative_to(repo)))
    if suspicious:
        R.p("Fichiers ou la normalisation semble ajustee sur l'ensemble complet :")
        for f in suspicious:
            R.p(f"- `{f}`  <- verifier manuellement")
    if good:
        R.p("\nFichiers avec ajustement explicitement train-only :")
        for f in good:
            R.p(f"- `{f}`")
    # PASS si au moins un chemin train-only et aucun suspect dans le pipeline extremes
    crit_sus = [s for s in suspicious if "exp_extreme" in s.replace("\\", "/")]
    R.check("E1", "Normalisation train-only",
            "PASS" if good and not crit_sus else ("FAIL" if crit_sus else ("PASS" if good else "INCONNU")))

    R.h("[F] Calibration — perimetre d'ajustement (pipeline extremes)")
    # train.py doit exporter RAW (pas d'affine) ; calibration.py + aggregate_seeds.py = bornée train-only
    cal = repo / "calibration.py"
    agg = repo / "aggregate_seeds.py"
    train = repo / "experiments/exp_extreme_train_intermediate_test_hybrid_lstm_transformer/train.py"
    if not cal.exists() or not agg.exists():
        R.check("F1", "Calibration bornee et train-only", "INCONNU",
                "calibration.py ou aggregate_seeds.py manquant")
        return
    t_cal = cal.read_text(encoding="utf-8", errors="ignore")
    t_agg = agg.read_text(encoding="utf-8", errors="ignore")
    t_tr = train.read_text(encoding="utf-8", errors="ignore") if train.exists() else ""
    bounded = ("arctanh" in t_cal or "atanh" in t_cal) and ("arctanh" in t_agg or "atanh" in t_agg or "fit_bounded" in t_agg)
    train_only = bool(re.search(
        r"(\[['\"]split['\"]\]\s*==\s*['\"]train['\"]|split\s*==\s*['\"]train['\"])",
        t_agg,
    ))
    # train.py ne doit PAS appliquer d'affine non bornée aux scores exportés
    train_applies_affine = bool(re.search(r"(polyfit|linregress|a\s*\*\s*x\s*\+\s*b)", t_tr)) and "raw_score" in t_tr
    R.p(f"- `calibration.py` : bornee={'oui' if ('arctanh' in t_cal or 'atanh' in t_cal) else 'non'}")
    R.p(f"- `aggregate_seeds.py` : train-only={'oui' if train_only else 'non'}, utilise calibration bornee={'oui' if bounded else 'non'}")
    R.p(f"- `train.py` extremes : exporte raw_score, pas d'affine={'oui' if not train_applies_affine else '**non**'}")
    ok = bounded and train_only and not train_applies_affine
    R.check("F1", "Calibration bornee et train-only",
            "PASS" if ok else "FAIL",
            "" if ok else "pipeline extremes : calibration manquante ou affine dans train.py")


# --------------------------------------------------- [G] convention temporelle
def temporal_convention(repo: Path, R: Report):
    R.h("[G] Convention temporelle et niveau d'agregation")
    hits = []
    for f in repo.rglob("*.py"):
        if _skip_py(f):
            continue
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if re.search(r"(temporal|phase|normalized_time|t_norm|Early|Middle|Late)", t):
            inv = bool(re.search(r"(\[::-1\]|invert_xaxis|flipud|reversed\()", t))
            lvl = ("participant" if re.search(r"groupby\(\s*\[?['\"]participant", t)
                   else "essai" if re.search(r"groupby\(\s*\[?['\"](trial|essai)", t)
                   else "?")
            hits.append((str(f.relative_to(repo)), inv, lvl))
    if not hits:
        R.check("G1", "Scripts temporels identifies", "INCONNU")
        return
    R.p("| script | inversion d'axe detectee | agregation |")
    R.p("|---|---|---|")
    for f, inv, lvl in hits:
        R.p(f"| `{f}` | {'**OUI**' if inv else 'non'} | {lvl} |")
    levels = {lvl for _, _, lvl in hits if lvl != "?"}
    R.check("G1", "Aucune inversion d'axe",
            "FAIL" if any(i for _, i, _ in hits) else "PASS")
    R.check("G2", "Niveau d'agregation unique",
            "PASS" if len(levels) <= 1 else "FAIL",
            f"niveaux melanges : {levels}" if len(levels) > 1 else "")


# --------------------------------------------------------- [H] reproductibilite
def reproducibility(repo: Path, R: Report):
    R.h("[H] Reproductibilite")
    git_hash = sh("git rev-parse HEAD")
    # on ignore le rapport lui-meme : il est ecrit par cet audit
    dirty = "\n".join(l for l in sh("git status --porcelain").splitlines()
                      if "PROVENANCE.md" not in l and not l.strip().endswith("docs/"))
    R.p(f"- commit : `{git_hash or 'INCONNU'}`")
    R.p(f"- arbre de travail : {'**modifie (non commite)**' if dirty else 'propre'}")
    R.check("H1", "Commit identifie", "PASS" if git_hash else "INCONNU")
    R.check("H2", "Arbre propre", "PASS" if not dirty else "FAIL",
            "impossible de rejouer un run depuis un arbre sale" if dirty else "")
    seeds = []
    for f in repo.rglob("*.py"):
        if _skip_py(f):
            continue
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if re.search(r"(manual_seed|random\.seed|set_seed|seed\s*=)", t):
            seeds.append(str(f.relative_to(repo)))
    R.p("\nFichiers fixant une seed : " + (", ".join(f"`{s}`" for s in seeds) or "_aucun_"))
    R.check("H3", "Seeds fixees", "PASS" if seeds else "FAIL")
    R.p("\n**Environnement**")
    for mod in ("numpy", "pandas", "scipy", "torch", "tensorflow", "sklearn"):
        try:
            m = __import__(mod)
            R.p(f"- {mod} : `{getattr(m,'__version__','?')}`")
        except Exception:
            R.p(f"- {mod} : _absent_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--scores-glob", default="runs/seed_*.csv")
    ap.add_argument("--out", default="docs/PROVENANCE.md")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    R = Report()
    R.p(f"Depot audite : `{repo}`\n")
    inventory(repo, R)
    tensor_version(repo, R)
    split_integrity(repo, R, a.scores_glob)
    norm_and_calib(repo, R)
    temporal_convention(repo, R)
    reproducibility(repo, R)
    out = repo / a.out
    R.dump(out)
    print(f"\nRapport ecrit : {out}\n")
    print(f"{'code':<6}{'verification':<44}statut")
    for c, l, s in R.verdicts:
        print(f"{c:<6}{l:<44}{s}")
    n_fail = sum(1 for _, _, s in R.verdicts if s == "FAIL")
    n_unk = sum(1 for _, _, s in R.verdicts if s == "INCONNU")
    print(f"\n{n_fail} ECHEC · {n_unk} INCONNU")
    if n_fail or n_unk:
        print("NE PAS LANCER LE RUN.")
        sys.exit(1)
    print("Pipeline verrouille.")


if __name__ == "__main__":
    main()
