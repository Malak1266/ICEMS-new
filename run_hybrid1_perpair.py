"""
run_hybrid1_perpair.py
======================
Orchestrateur : entrainement 10 modeles per-paire + evaluation composite.

Usage :
    python run_hybrid1_perpair.py --data v1 --seed 42
    python run_hybrid1_perpair.py --data v2 --seed 42
    python run_hybrid1_perpair.py --both --seed 42
    python run_hybrid1_perpair.py --data v1 --seed 42 --train-only
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from data_hybrid1 import resolve_hybrid1_paths

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_hybrid1_perpair")

_CUDA_OOM_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUDNN_STATUS_ALLOC_FAILED",
    "cuda runtime error",
)


def _is_cuda_oom(output: str) -> bool:
    text = output.lower()
    return any(marker.lower() in text for marker in _CUDA_OOM_MARKERS)


def run_cmd(cmd: list[str]) -> None:
    logger.info(f"[exec] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_train_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    logger.info(f"[exec] {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)


def train_with_oom_fallback(
    data: str | None,
    pkl: Path,
    out: Path,
    seed: int,
    max_len: int,
    batch_size: int,
    *,
    butterworth: bool,
    pairs: list[int] | None = None,
) -> int:
    """Lance l'entrainement ; reduit batch_size de moitie UNIQUEMENT sur CUDA OOM."""
    bs = batch_size
    while bs >= 1:
        cmd = [
            sys.executable, "train_hybrid1_perpair.py",
            "--out", str(out),
            "--seed", str(seed),
            "--max-len", str(max_len),
            "--batch-size", str(bs),
        ]
        if data is not None:
            cmd.extend(["--data", data])
        else:
            cmd.extend(["--pkl", str(pkl)])
        if butterworth:
            cmd.append("--butterworth")
        else:
            cmd.append("--no-butterworth")
        if pairs is not None:
            cmd.extend(["--pairs", *[str(p) for p in pairs]])

        result = _run_train_capture(cmd)
        combined = (result.stdout or "") + (result.stderr or "")

        if result.returncode == 0:
            if result.stdout:
                sys.stdout.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    sys.stdout.write("\n")
            return bs

        if result.stdout:
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
            if not result.stderr.endswith("\n"):
                sys.stderr.write("\n")

        if _is_cuda_oom(combined):
            if bs <= 1:
                raise RuntimeError(
                    "CUDA out of memory meme avec batch_size=1 — "
                    "impossible de continuer l'entrainement."
                ) from None
            logger.warning(
                f"[OOM] batch={bs} — CUDA out of memory, retry batch={bs // 2}"
            )
            bs //= 2
            continue

        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr,
        )

    raise RuntimeError("train_with_oom_fallback : etat inattendu")


def run_one(
    data: str,
    args: argparse.Namespace,
    pairs: list[int] | None = None,
) -> None:
    pkl, out = resolve_hybrid1_paths(data, args.pkl, args.out)
    if not pkl.exists():
        raise FileNotFoundError(f"Donnees manquantes : {pkl}")

    logger.info("=" * 70)
    logger.info(f"PIPELINE Hybrid1 per-pair — data={data}")
    logger.info("=" * 70)
    logger.info(f"seed={args.seed} max_len={args.max_len} pkl={pkl} out={out}")

    if not args.eval_only:
        used_bs = train_with_oom_fallback(
            data, pkl, out, args.seed, args.max_len, args.batch_size,
            butterworth=args.butterworth,
            pairs=pairs,
        )
        logger.info(f"[train] termine data={data} (batch effectif={used_bs})")

    if not args.train_only:
        cmd = [
            sys.executable, "eval_hybrid1_perpair.py",
            "--data", data,
            "--seed", str(args.seed),
        ]
        if args.butterworth:
            cmd.append("--butterworth")
        else:
            cmd.append("--no-butterworth")
        run_cmd(cmd)
        logger.info(f"[eval] termine data={data}")

    logger.info(f"[done] resultats dans {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline Hybrid1 per-pair complet")
    ap.add_argument("--data", choices=["v1", "v2"], default=None,
                    help="Version trial_tensor (fixe pkl + out)")
    ap.add_argument("--both", action="store_true",
                    help="Enchaine v1 puis v2 puis compare_hybrid1_v1_v2.py")
    ap.add_argument("--pkl", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--butterworth", action=argparse.BooleanOptionalAction, default=False,
                    help="Defaut False : v1/v2 deja filtres en amont (--no-butterworth)")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--pairs", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.both:
        for ver in ("v1", "v2"):
            run_one(ver, args, pairs=args.pairs)
        run_cmd([sys.executable, "compare_hybrid1_v1_v2.py"])
        return

    if args.data is None:
        pkl, out = resolve_hybrid1_paths(None, args.pkl, args.out)
        if not pkl.exists():
            raise FileNotFoundError(f"Donnees manquantes : {pkl}")
        if not args.eval_only:
            train_with_oom_fallback(
                None, pkl, out, args.seed, args.max_len, args.batch_size,
                butterworth=args.butterworth,
                pairs=args.pairs,
            )
        if not args.train_only:
            cmd = [
                sys.executable, "eval_hybrid1_perpair.py",
                "--pkl", str(pkl),
                "--run", str(out),
                "--seed", str(args.seed),
            ]
            if args.butterworth:
                cmd.append("--butterworth")
            else:
                cmd.append("--no-butterworth")
            run_cmd(cmd)
        logger.info(f"[done] resultats dans {out}")
        return

    run_one(args.data, args, pairs=args.pairs)


if __name__ == "__main__":
    main()
