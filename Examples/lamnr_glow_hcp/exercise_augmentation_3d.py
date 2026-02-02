#!/usr/bin/env python3
"""
Exercise 3D data augmentation for the HCP T1/T2/FA cohort.

- Uses the same build_loaders_from_globs_3d + ImageDataset pipeline
  as train_cohort_screened_3d.py
- Applies the augmentation (including schedules, if given)
- Writes augmented 3D volumes per view to disk as NIfTI files

If --step >= 0, we fix the effective "training step" for augmentation
schedules to that value. If --step < 0, we advance the schedule with
each batch (step = 1, 2, 3, ...).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import ants

from train_cohort_screened_3d import (
    build_loaders_from_globs_3d,
    to01,
    _extract_views_from_batch,
)


def main():
    ap = argparse.ArgumentParser("Exercise 3D ANTsTorch data augmentation")
    ap.add_argument(
        "--view",
        action="append",
        nargs="+",
        required=True,
        help=(
            "Repeat per view, same semantics as train_cohort_screened_3d.py. "
            "Each --view takes one or more glob patterns."
        ),
    )
    ap.add_argument("--H", type=int, default=64, help="Resampled height")
    ap.add_argument("--W", type=int, default=64, help="Resampled width")
    ap.add_argument("--D", type=int, default=64, help="Resampled depth")

    ap.add_argument("--train-samples", type=int, default=256,
                    help="number_of_samples passed to ImageDataset")
    ap.add_argument("--val-samples", type=int, default=1,
                    help="dummy; kept for API compatibility")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--subject-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--aug-schedules",
        type=str,
        default=(
            "noise_std:cos:0.05->0.00@150000,"
            "sd_affine:linear:0.05->0.00@80000,"
            "sd_deformation:cos:0.20->0.00@100000,"
            "sd_simulated_bias_field:cos:1.00->0.00@120000,"
            "sd_histogram_warping:exp:0.05->0.00@120000"
        ),
        help="Same multi-parameter anneal spec used in the trainer.",
    )
    ap.add_argument("--disable-aug-anneal", action="store_true")

    ap.add_argument(
        "--n-per-view",
        type=int,
        default=32,
        help="How many augmented volumes to save per view.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for augmented NIfTI images.",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="aug",
        help="Filename prefix for written volumes.",
    )

    ap.add_argument(
        "--step",
        type=int,
        default=-1,
        help=(
            "If >= 0, fix augmentation schedules to this effective training "
            "step for all volumes. If < 0, advance step with each batch "
            "(step = 1, 2, 3, ...)."
        ),
    )

    args = ap.parse_args()
    num_views = len(args.view)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- build dataset/loaders using the trainer helper (same augmentation path) ---
    train_loader, _, global_step = build_loaders_from_globs_3d(
        view_specs=args.view,
        H=args.H,
        W=args.W,
        D=args.D,
        train_samples=max(args.train_samples, args.n_per_view),
        val_samples=args.val_samples,
        batch=args.batch,
        num_workers=args.num_workers,
        val_frac=float(args.val_frac),
        subject_limit=(args.subject_limit if args.subject_limit > 0 else None),
        do_aug=True,
        aug_schedules=(
            args.aug_schedules if not args.disable_aug_anneal else None
        ),
        disable_aug_anneal=args.disable_aug_anneal,
        seed=args.seed,
    )

    per_view_counts = [0 for _ in range(num_views)]
    target_total = args.n_per_view * num_views

    if args.step >= 0:
        print(f"[info] using fixed augmentation step = {args.step}")
    else:
        print("[info] using batch index as augmentation step (1, 2, 3, ...)")

    print(f"[info] saving up to {args.n_per_view} augmented volumes per view")
    print(f"[info] output dir: {out_dir}")

    for it, batch in enumerate(train_loader, start=1):
        # Set effective "training step" for the augmentation scheduler.
        if args.step >= 0:
            step_val = args.step
        else:
            step_val = it

        with global_step.get_lock():
            global_step.value = step_val

        xs = _extract_views_from_batch(batch, num_views=num_views)

        for vi in range(num_views):
            if per_view_counts[vi] >= args.n_per_view:
                continue

            x_v = xs[vi]  # (B,1,D,H,W) or (B,D,H,W)
            x_v = to01(x_v)
            if x_v.ndim == 4:  # (B,D,H,W) -> add channel
                x_v = x_v.unsqueeze(1)

            B = x_v.shape[0]
            for b in range(B):
                if per_view_counts[vi] >= args.n_per_view:
                    break

                vol = x_v[b, 0].cpu().numpy().astype(np.float32)  # (D,H,W)
                img = ants.from_numpy(vol)

                out_path = out_dir / f"{args.prefix}_view{vi}_idx{per_view_counts[vi]:04d}.nii.gz"
                ants.image_write(img, str(out_path))
                per_view_counts[vi] += 1

        if sum(per_view_counts) >= target_total:
            break

    print(f"[done] per-view counts: {per_view_counts}")
    print(f"[done] augmented volumes written to {out_dir}")


if __name__ == "__main__":
    main()
