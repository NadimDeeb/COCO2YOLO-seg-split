#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025 Nadim Dib ngd04@mail.aub.edu

"""
Train/Val/Test folder splitter with reproducible seeds.

Examples:
  split_tvt.py --input /data/yolo --output /data/yolo_split --ratio 0.8 0.1 0.1 --seed 42
  split_tvt.py -i images_labels -o split_out --ratio 0.8 0.2 --move

Notes:
- Expects your input root to contain subfolders like `images/` and `labels/`
  so paired files stay together (split-folders groups by subdir).
"""

import argparse
import os
import sys
from typing import List, Tuple

def _parse_ratio(vals: List[float]) -> Tuple[float, float, float]:
    """
    Accepts 2 or 3 floats. If they don't sum to 1, normalize them.
    Returns a 3-tuple (train, val, test).
    """
    if not vals:
        vals = [0.8, 0.1, 0.1]
    if len(vals) not in (2, 3):
        print("ERROR: --ratio must have 2 or 3 numbers (train val [test])", file=sys.stderr)
        sys.exit(2)
    if len(vals) == 2:
        vals = [vals[0], vals[1], 0.0]
    s = sum(vals)
    if s <= 0:
        print("ERROR: Sum of ratios must be > 0", file=sys.stderr)
        sys.exit(2)
    if abs(s - 1.0) > 1e-6:
        vals = [v / s for v in vals]
        print(f"[INFO] Ratios normalized to sum=1.0 -> {tuple(round(v, 4) for v in vals)}", file=sys.stderr)
    return float(vals[0]), float(vals[1]), float(vals[2])

def main():
    try:
        import splitfolders
    except Exception as e:
        print("ERROR: split-folders is required. Install with: pip install split-folders", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser(description="Split a dataset directory into train/val[/test].")
    p.add_argument("-i", "--input", required=True, help="Input root (contains subfolders like images/, labels/)")
    p.add_argument("-o", "--output", required=True, help="Output root to create train/ val/ (and test/)")
    p.add_argument("--ratio", nargs="+", type=float, default=[0.8, 0.1, 0.1],
                   help="Proportions: train val [test]. Defaults to 0.8 0.1 0.1")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    p.add_argument("--group-prefix", default=None,
                   help="Group files by a common prefix so they stay together (pass a prefix string)")
    p.add_argument("--move", action="store_true",
                   help="Move files instead of copying (default copies)")
    args = p.parse_args()

    train, val, test = _parse_ratio(args.ratio)

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    # Build the ratio tuple expected by split-folders: 2 or 3 numbers
    ratio_tuple = (train, val) if test == 0.0 else (train, val, test)

    print(f"[INFO] Splitting\n  input : {args.input}\n  output: {args.output}\n"
          f"  ratio : {ratio_tuple}\n  seed  : {args.seed}\n"
          f"  move  : {args.move}\n  group : {args.group_prefix}", file=sys.stderr)

    # Perform the split
    splitfolders.ratio(
        input=args.input,
        output=args.output,
        seed=args.seed,
        ratio=ratio_tuple,
        group_prefix=args.group_prefix,
        move=args.move,
    )

    print("[DONE] Split complete.")

if __name__ == "__main__":
    main()
import splitfolders

splitfolders.ratio(
    input="/mnt/haus/Downloads/cococo",    # folder with sub-folders images/ and labels/
    output="/mnt/haus/Downloads/cocout",   # new root for train/val/test
    seed=42,                           # your chosen seed
    ratio=(.8, .2, .2),                # train/val/test proportions
    group_prefix=None                  # None means split files by name across sub-dirs
)

