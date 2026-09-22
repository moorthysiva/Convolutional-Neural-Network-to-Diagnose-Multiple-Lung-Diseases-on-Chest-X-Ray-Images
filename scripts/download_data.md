# Getting the NIH ChestX-ray14 data

The dataset is ~42 GB and is **not** bundled with this repo. Download it once and
lay it out under `data/`.

## Option A — Kaggle (easiest)

```bash
pip install kaggle
kaggle datasets download -d nih-chest-xrays/data -p data/ --unzip
```

## Option B — NIH Box (official)

Download from the NIH release:
<https://nihcc.app.box.com/v/ChestXray-NIHCC>

Unpack each `images_XXX.tar.gz` into its own folder.

## Expected layout

```
data/
├── Data_Entry_2017.csv        # metadata + labels
├── train_val_list.txt         # official patient-disjoint split (recommended)
├── test_list.txt
├── images_001/  ... images_012/   # the .png files
```

If `train_val_list.txt` / `test_list.txt` are present the pipeline uses the
official split automatically; otherwise it falls back to a grouped split on
`Patient ID` (still leakage-free).

> **Data governance:** these are de-identified public research images. If you
> ever adapt this to real clinical data, treat images as PHI — encrypt at rest,
> restrict access, and keep them out of version control (`data/` is gitignored).
