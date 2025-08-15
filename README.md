# COCO2YOLO-seg-split

Tools for preparing YOLO **segmentation** datasets:

- **COCO → YOLO segmentation** converter: `coco2yolo_seg.py`  
  Handles **polygons** and **RLE** (“crowd”) masks, safe normalization, optional polygon simplification.
- **Train/Val/Test splitter**: `split_tvt.py`  
  Reproducible splits with a fixed seed; supports 2-way or 3-way ratios.

> This converter is **derived from** [coco2yolo](https://github.com/tw-yshuang/coco2yolo) and is distributed under **GPL-3.0**. The splitter may be MIT, but the combined repository is effectively GPL due to the converter.

---

## Install

**Python:** 3.8+

**Dependencies**
```bash
pip install numpy opencv-python split-folders
# Optional (for RLE support):
pip install pycocotools                 # typical x86_64
# Jetson/ARM:
pip install pycocotools-linux-aarch64
```

---

## Quickstart

```bash
# 1) Convert COCO → YOLO segmentation
python3 coco2yolo_seg.py   --json /path/instances.json   --images_root /path/images   --out /path/yolo_labels

# 2) Split into train/val/test
python3 split_tvt.py   --input /path/dataset_root   --output /path/split_out   --ratio 0.8 0.1 0.1   --seed 42
```

---

## COCO → YOLO Segmentation (`coco2yolo_seg.py`)

**Usage**
```bash
python3 coco2yolo_seg.py   --json /path/to/annotations.json   --images_root /path/to/images_dir   --out /path/to/output_labels   [--skip-crowd]   [--approx-epsilon 1.5]   [--min-points 3]
```

**Arguments**
- `--json` (required): COCO **instances** JSON.
- `--images_root` (required): Directory containing all images referenced by `file_name` in the JSON.
- `--out` (required): Output directory for YOLO `*.txt` labels (created if missing).
- `--skip-crowd`: Skip `iscrowd==1` RLE annotations (useful if you don’t want RLE or don’t have `pycocotools`).
- `--approx-epsilon`: Polygon simplification epsilon in **pixels** (0 = off). Typical values: `1.0–2.0`.
- `--min-points`: Minimum polygon vertices to keep (default `3`).

**Output format (one line per polygon)**
```
<class_id> x1 y1 x2 y2 ... xN yN
```
- Coordinates are **normalized** to `[0,1]`.
- Polygons with fewer than 3 points are dropped.
- If an image ends up with no valid polygons, its label file is removed.

**Class mapping**
- COCO `category_id`s are remapped to `0..N-1` by **sorted category id** order. Keep a mapping table if you need names.

**Notes**
- If `width`/`height` are missing in the JSON, the script tries to read the image to infer them.
- With RLE:
  - Requires `pycocotools` (or `pycocotools-linux-aarch64` on Jetson).
  - Multiple contours from one RLE mask produce **multiple lines** (same class).

**Examples**
```bash
# Basic conversion
python3 coco2yolo_seg.py   --json data/instances.json   --images_root data/images   --out labels

# Skip crowds and simplify polygons by ~1.5 px
python3 coco2yolo_seg.py   --json data/instances.json   --images_root data/images   --out labels   --skip-crowd --approx-epsilon 1.5
```

---

## Train/Val/Test Split (`split_tvt.py`)

**Usage**
```bash
python3 split_tvt.py   --input /path/dataset_root   --output /path/split_out   --ratio 0.8 0.1 0.1   --seed 42   [--group-prefix <prefix>]   [--move]
```

**Flags**
- `--input`: Dataset root that contains subfolders like `images/`, `labels/`.
- `--output`: Output root; creates `train/`, `val/` (and optionally `test/`).
- `--ratio`: Two or three numbers: `train val [test]`. If they don’t sum to 1, they are normalized.
- `--seed`: Fixed seed for deterministic splits.
- `--group-prefix`: Keep files that share a common prefix together (optional).
- `--move`: Move files instead of copying.

**Examples**
```bash
# 80/10/10 split (copy)
python3 split_tvt.py -i data -o data_split --ratio 0.8 0.1 0.1 --seed 42

# 90/10 split (no test)
python3 split_tvt.py -i data -o data_split --ratio 0.9 0.1 --seed 123
```

---

## Folder Layouts

**Before (COCO)**
```
data/
├─ images/
│  ├─ 0001.jpg
│  ├─ 0002.jpg
│  └─ ...
└─ instances.json
```

**After conversion (YOLO labels)**
```
labels/
├─ 0001.txt
├─ 0002.txt
└─ ...
```

**After splitting**
```
split_out/
├─ train/
│  ├─ images/
│  └─ labels/
├─ val/
│  ├─ images/
│  └─ labels/
└─ test/              # only if 3-way split
   ├─ images/
   └─ labels/
```

---

## FAQ / Troubleshooting

- **`ModuleNotFoundError: cv2`** → `pip install opencv-python`
- **RLE isn’t processed / “pycocotools” missing** → `pip install pycocotools` (x86_64) or `pip install pycocotools-linux-aarch64` (Jetson/ARM), or run with `--skip-crowd`.
- **Some images don’t get a label file** → all polygons were invalid (too few points) or filtered; empty label files are removed by design.
- **Do I need to close the polygon by repeating the first point?** → No. YOLOv5/8 segmentation accepts the vertex list without repeating the start point.
- **Which class id maps to which name?** → We remap by sorted COCO `category_id`. Keep a mapping table if you need to reference names later.

---

## Contributing

PRs welcome! If you contribute code to the converter, it must be **GPL-3.0** compatible.  
For the splitter, MIT contributions are fine—but the repository is distributed under GPL because of the converter.

---

## License

- **Repository:** GNU General Public License v3.0 (**GPL-3.0**). See `LICENSE`.
- **split_tvt.py:** may be released under **MIT** (see `LICENSES/MIT.txt`), but distribution with GPL code means the repo is effectively GPL.

---

## Attribution

- Derived from **[coco2yolo](https://github.com/tw-yshuang/coco2yolo)** (GPL-3.0).
- Additional features added here: RLE (crowd) decoding, YOLOv5/8 segmentation output, polygon simplification & bound checks.
