#!/usr/bin/env python3
"""
Sweep coordinator for single-view tabular LAMNr flows.

This launcher is compatible with train_lamnr_flows_tabular.py, which supports:
  --output-prefix <path/prefix>
  --save-checkpoint-dir <dir>
  --lambda-penalty <float>

Older sweepers used flags not supported by the trainer:
  --outdir, --checkpoints-dir, --alignment-lambda

This script keeps the *sweeper* interface (outdir/manifest) but maps to trainer args:

  run_dir = <outdir>/<output_prefix>_K{K}_hc{hc}
  trainer --output-prefix         = <run_dir>/out
  trainer --save-checkpoint-dir   = <run_dir>/checkpoints
  trainer --lambda-penalty        = manifest alignment_lambda (or lambda_penalty)

Expected outputs per run_dir:
  out_whitened_view0.csv, out_whitened_view1.csv, ...
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _parse_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    if not rows:
        raise SystemExit(f"[ERR] Empty manifest: {path}")
    return rows


def _as_float(x: str, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise SystemExit(f"[ERR] Cannot parse float for {name}='{x}': {e}")


def _split_list(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    parts: List[str] = []
    for chunk in s.split(","):
        parts.extend(chunk.strip().split())
    return [p for p in parts if p]


def _exists_whitened_view0(run_dir: Path) -> bool:
    if (run_dir / "out_whitened_view0.csv").exists():
        return True
    return len(list(run_dir.glob("*_whitened_view0.csv"))) > 0


def _print_cmd(cmd: Sequence[str]) -> None:
    print("CMD:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print()


@dataclass
class Job:
    cmd: List[str]
    run_dir: Path
    label: str


def build_job(
    *,
    python_exe: str,
    trainer: Path,
    views: List[str],
    cuda_device: str,
    screening_mode: str,
    outdir: Path,
    row: Dict[str, str],
    K: int,
    hidden_channels: int,
    extra: List[str],
) -> Job:
    output_prefix_base = str(row.get("output_prefix", "")).strip()
    if not output_prefix_base:
        raise SystemExit("[ERR] manifest must include 'output_prefix' column")

    rel = f"{output_prefix_base}_K{K}_hc{hidden_channels}"
    run_dir = outdir / rel
    base_prefix = run_dir / "out"

    # Map older manifest field alignment_lambda -> trainer lambda-penalty
    lambda_penalty: Optional[float] = None
    if str(row.get("lambda_penalty", "")).strip() != "":
        lambda_penalty = _as_float(str(row["lambda_penalty"]), "lambda_penalty")
    elif str(row.get("alignment_lambda", "")).strip() != "":
        lambda_penalty = _as_float(str(row["alignment_lambda"]), "alignment_lambda")

    cmd: List[str] = [
        python_exe,
        str(trainer),
        "--views",
        *views,
        "--cuda-device",
        cuda_device,
        "--screening-mode",
        screening_mode,
        "--output-prefix",
        str(base_prefix),
        "--save-checkpoint-dir",
        str(run_dir / "checkpoints"),
        "--K",
        str(K),
        "--hidden-channels",
        str(hidden_channels),
        "--verbose",
        ]

    def add_if_present(flag: str, key: str) -> None:
        v = row.get(key, "")
        if v is None:
            return
        v = str(v).strip()
        if v == "":
            return
        cmd.extend([flag, v])

    add_if_present("--base-distribution", "base_distribution")
    add_if_present("--pca-latent-dimension", "pca_latent_dimension")
    add_if_present("--base-sigma", "base_sigma")
    add_if_present("--normalization", "normalization")
    add_if_present("--impute", "impute")
    add_if_present("--add-noise-in", "add_noise_in")
    add_if_present("--jitter-alpha", "jitter_alpha")
    add_if_present("--jitter-alpha-end", "jitter_alpha_end")
    add_if_present("--jitter-alpha-mode", "jitter_alpha_mode")
    add_if_present("--jitter-alpha-total-steps", "jitter_alpha_total_steps")
    add_if_present("--lr", "lr")
    add_if_present("--weight-decay", "weight_decay")
    add_if_present("--max-iter", "max_iter")
    add_if_present("--batch-size", "batch_size")
    add_if_present("--val-batch-size", "val_batch_size")
    add_if_present("--tradeoff-mode", "tradeoff_mode")
    add_if_present("--penalty-type", "penalty_type")
    add_if_present("--best-selection-metric", "best_selection_metric")
    add_if_present("--seed", "seed")

    if lambda_penalty is not None:
        cmd.extend(["--lambda-penalty", str(lambda_penalty)])

    if extra:
        cmd.extend(extra)

    return Job(cmd=cmd, run_dir=run_dir, label=rel)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--trainer", default="../train_lamnr_flows_tabular.py")
    ap.add_argument("--python", dest="python_exe", default=sys.executable)
    ap.add_argument("--views", required=True, nargs="+")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cuda-device", default="cuda:0")
    ap.add_argument("--screening-mode", default="none", choices=["none", "hsic", "cca"])
    ap.add_argument("--K-sweep", required=True)
    ap.add_argument("--hidden-sweep", required=True)
    ap.add_argument("--max-procs", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    args = ap.parse_args(argv)

    rows = _parse_csv(Path(args.manifest))

    K_list = [int(x) for x in _split_list(args.K_sweep)]
    hc_list = [int(x) for x in _split_list(args.hidden_sweep)]
    if not K_list:
        raise SystemExit("[ERR] --K-sweep is empty")
    if not hc_list:
        raise SystemExit("[ERR] --hidden-sweep is empty")

    trainer = Path(args.trainer)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs: List[Job] = []
    for row in rows:
        for K in K_list:
            for hc in hc_list:
                j = build_job(
                    python_exe=args.python_exe,
                    trainer=trainer,
                    views=args.views,
                    cuda_device=args.cuda_device,
                    screening_mode=args.screening_mode,
                    outdir=outdir,
                    row=row,
                    K=K,
                    hidden_channels=hc,
                    extra=args.extra,
                )
                if args.skip_existing and _exists_whitened_view0(j.run_dir):
                    print(f"SKIP (exists): {j.run_dir}")
                    continue
                jobs.append(j)

    print(f"Prepared {len(jobs)} jobs (K={len(K_list)} × hc={len(hc_list)} × rows={len(rows)}).")

    if args.dry_run:
        for j in jobs:
            _print_cmd(j.cmd)
        return 0

    running: List[Tuple[subprocess.Popen, Job]] = []
    idx = 0

    def _wait_one() -> None:
        nonlocal running
        p, j = running[0]
        rc = p.wait()
        running = running[1:]
        if rc != 0:
            raise RuntimeError(f"Trainer exited with code {rc} for {j.label}")

    while idx < len(jobs) or running:
        while idx < len(jobs) and len(running) < args.max_procs:
            j = jobs[idx]
            idx += 1
            j.run_dir.mkdir(parents=True, exist_ok=True)
            _print_cmd(j.cmd)
            p = subprocess.Popen(j.cmd)
            running.append((p, j))
            time.sleep(0.2)

        if running:
            _wait_one()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
