#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO -> YOLO segmentation label converter
- Supports polygon and RLE (crowd) segmentations
- Produces YOLOv5/8-style segmentation labels:
  each line: <class_id> x1 y1 x2 y2 ... (normalized)
- One .txt per image in an output labels directory

Usage:
  python3 coco2yolo_seg.py \
      --json /path/to/annotations.json \
      --images_root /path/to/images_dir \
      --out /path/to/output_labels

Notes:
- Requires: opencv-python, numpy
- For RLE support: pycocotools (or pycocotools-linux-aarch64 on Jetson)
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV (cv2) is required. Install with 'pip install opencv-python'.") from e

# Try to import pycocotools for RLE; if missing we can skip crowds
try:
    from pycocotools import mask as maskUtils  # type: ignore
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _contour_to_xy(contour: np.ndarray) -> np.ndarray:
    """
    contour: (N,1,2) int or float
    returns (N,2) float array [ [x,y], ... ]
    """
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)
    return contour.astype(np.float32)


def _normalize_xy(xy: np.ndarray, w: int, h: int) -> np.ndarray:
    # clip to image bounds then normalize
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1) / max(w, 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1) / max(h, 1)
    return xy


def _approx_simplify(xy: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Simplify polygon with Douglas-Peucker (epsilon in pixels, before normalization).
    If epsilon==0, returns as-is.
    """
    if epsilon <= 0 or xy.shape[0] < 4:
        return xy
    cnt = xy.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx = approx.reshape(-1, 2)
    return approx if approx.shape[0] >= 3 else xy


def _write_label_line(fh, cls_id: int, xy_norm: np.ndarray):
    """
    Write one YOLO segmentation line: <class> x1 y1 x2 y2 ...
    Requires at least 3 points (>=6 numbers).
    """
    if xy_norm.shape[0] < 3:
        return
    flat = []
    for x, y in xy_norm:
        flat.append(f"{x:.5f}")
        flat.append(f"{y:.5f}")
    fh.write(f"{cls_id} " + " ".join(flat) + "\n")


def _rle_to_contours(rle, w: int, h: int):
    """
    Decode an RLE mask to external contours using OpenCV.
    Returns list of contours (each (N,1,2) numpy array).
    """
    if not HAS_PYCOCO:
        return []
    m = maskUtils.decode(rle)  # HxWx1 or HxW
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m > 0).astype(np.uint8)  # 0/1
    # Find external contours
    res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 3:
        _, contours, _ = res
    else:
        contours, _ = res
    return contours


def convert(
    json_file: str,
    images_root: str,
    out_dir: str,
    skip_crowd: bool = False,
    approx_epsilon: float = 0.0,
    min_points: int = 3,
):
    """
    json_file: path to instances/annotations json
    images_root: directory where image file_name(s) live
    out_dir: output directory to write labels/*.txt
    skip_crowd: if True, skip annotations with iscrowd==1 (RLE)
    approx_epsilon: contour simplification epsilon (in pixels, before normalization)
    min_points: minimum polygon points to keep (>=3)
    """
    with open(json_file, "r") as f:
        coco = json.load(f)

    # Build maps for quick lookup
    img_by_id = {im["id"]: im for im in coco["images"]}
    cat_ids = sorted([c["id"] for c in coco["categories"]])
    catid2yolo = {cid: i for i, cid in enumerate(cat_ids)}

    # Group annotations by image
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    labels_dir = out_dir
    _ensure_dir(labels_dir)

    # Stats
    total_imgs = len(img_by_id)
    written_imgs = 0
    total_anns = 0
    written_lines = 0
    skipped_rle = 0
    skipped_empty = 0

    for img_id, img in img_by_id.items():
        file_name = img.get("file_name")
        if not file_name:
            continue
        w = int(img.get("width", 0))
        h = int(img.get("height", 0))
        if w <= 0 or h <= 0:
            # try to read actual image to infer size if needed
            full_path = os.path.join(images_root, file_name)
            if os.path.isfile(full_path):
                im = cv2.imread(full_path)
                if im is not None:
                    h, w = im.shape[:2]

        label_path = os.path.join(
            labels_dir, os.path.splitext(os.path.basename(file_name))[0] + ".txt"
        )
        lines_written_for_image = 0

        with open(label_path, "w") as fh:
            for ann in anns_by_img.get(img_id, []):
                total_anns += 1
                cat_id = ann["category_id"]
                cls_id = catid2yolo[cat_id]

                seg = ann.get("segmentation", None)
                if not seg:
                    skipped_empty += 1
                    continue

                # Handle polygon lists
                if isinstance(seg, list):
                    # seg could be multi-poly (list of lists)
                    for poly in seg:
                        if not poly or len(poly) < 6:
                            continue
                        xy = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        xy = _approx_simplify(xy, approx_epsilon)
                        if xy.shape[0] < min_points:
                            continue
                        xy = _normalize_xy(xy, w, h)
                        _write_label_line(fh, cls_id, xy)
                        written_lines += 1
                        lines_written_for_image += 1

                # Handle RLE dict
                elif isinstance(seg, dict):
                    iscrowd = int(ann.get("iscrowd", 0))
                    if iscrowd == 1 and skip_crowd:
                        skipped_rle += 1
                        continue
                    if not HAS_PYCOCO:
                        # cannot decode without pycocotools
                        skipped_rle += 1
                        continue
                    # Ensure proper RLE format
                    rle = seg
                    if "counts" in rle and isinstance(rle["counts"], list):
                        # uncompressed RLE; convert to RLE
                        rle = maskUtils.frPyObjects(rle, h, w)
                    contours = _rle_to_contours(rle, w, h)
                    if not contours:
                        skipped_empty += 1
                        continue
                    for cnt in contours:
                        xy = _contour_to_xy(cnt)
                        xy = _approx_simplify(xy, approx_epsilon)
                        if xy.shape[0] < min_points:
                            continue
                        xy = _normalize_xy(xy, w, h)
                        _write_label_line(fh, cls_id, xy)
                        written_lines += 1
                        lines_written_for_image += 1

                else:
                    # Unknown segmentation format
                    skipped_empty += 1
                    continue

        # If no lines written, remove empty file
        if lines_written_for_image == 0 and os.path.exists(label_path):
            os.remove(label_path)
        else:
            written_imgs += 1

    print(f"[DONE]")
    print(f" Images total:   {total_imgs}")
    print(f" Images labeled: {written_imgs}")
    print(f" Annotations:    {total_anns}")
    print(f" YOLO lines:     {written_lines}")
    print(f" Skipped RLE:    {skipped_rle} (missing pycocotools or --skip-crowd)")
    print(f" Skipped empty:  {skipped_empty}")


def parse_args():
    ap = argparse.ArgumentParser(description="COCO -> YOLO segmentation converter")
    ap.add_argument("--json", required=True, help="Path to COCO annotations JSON")
    ap.add_argument(
        "--images_root",
        required=True,
        help="Directory containing the images referenced by file_name",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output directory for YOLO labels (.txt). Will be created if missing.",
    )
    ap.add_argument(
        "--skip-crowd",
        action="store_true",
        help="Skip iscrowd==1 (RLE) annotations",
    )
    ap.add_argument(
        "--approx-epsilon",
        type=float,
        default=0.0,
        help="Douglas-Peucker simplification epsilon in pixels (0 = off)",
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum polygon points to keep (>=3)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        json_file=args.json,
        images_root=args.images_root,
        out_dir=args.out,
        skip_crowd=args.skip_crowd,
        approx_epsilon=args.approx_epsilon,
        min_points=args.min_points,
    )
