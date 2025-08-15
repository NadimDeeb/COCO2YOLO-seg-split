# COCO2YOLO-seg-split

Tools for preparing YOLO segmentation datasets:

- **COCO → YOLO segmentation** converter (`coco2yolo_seg.py`)
  - Polygons **and** RLE (“crowd”) masks (optional `pycocotools`)
  - Optional polygon simplification
  - Safe normalization & bound clipping
- **Train/Val/Test splitter** (`split_tvt.py`)
  - Reproducible splits with a fixed seed

> This repo’s converter is **derived from** [coco2yolo](https://github.com/tw-yshuang/coco2yolo) and distributed under **GPL-3.0**. The splitter may be MIT, but the combined repo remains GPL due to the converter.

---

## Contents

- [Features](#features)
- [Install](#install)
- [Quickstart](#quickstart)
- [1) COCO → YOLO Segmentation](#1-coco--yolo-segmentation)
- [2) Train/Val/Test Split](#2-trainvaltest-split)
- [Folder Layouts](#folder-layouts)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Attribution](#attribution)

---

## Features

**Converter (`coco2yolo_seg.py`)**
- Converts COCO **segmentation** to YOLOv5/8 `*.txt` labels
- Supports:
  - **Polygons** (`segmentation: [x1,y1,x2,y2,...]`)
  - **RLE** masks (`segmentation: {counts, size}`) when `pycocotools` is installed
- Optional **Douglas–Peucker** simplification via `--approx-epsilon`
- Clips coordinates to image bounds; normalizes to `[0,1]`
- Drops degenerate polygons (`--min-points` ≥ 3)

**Splitter (`split_tvt.py`)**
- Deterministic splits with `--seed`
- Supports 2-way (`train/val`) or 3-way (`train/val/test`) ratios
- Copies by default; `--move` to move files

---

## Install

**Python:** 3.8+

**Dependencies**
```bash
pip install numpy opencv-python split-folders
# Optional (RLE support):
pip install pycocotools             # typical x86_64
# Jetson/ARM:
pip install pycocotools-linux-aarch64
