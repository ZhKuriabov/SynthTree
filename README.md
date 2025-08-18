# SynthTree — README

This repository contains Jupyter notebooks for two projects: **SKCM** and **BikeSharing**. Both notebooks provide a detailed step-by-step implementation of the **SynthTree** method together with **MLM-EPIC** method. The main entry points are ```SKCM.ipynb```,  ```BikeSharing.ipynb```,  and ```co_supervision_test.py```, which automates: (1) clustering & co-supervision augmentation (2) SynthTree construction, and (3) training and evaluation with summary exports.

---

## Contents

### Notebooks
- **BikeSharing.ipynb** — This notebook illustrates the application of the SynthTree method within the BikeSharing dataset, guiding users through the implementation process and analysis.
- **SKCM.ipynb** — This notebook illustrates the application of the SynthTree method within the SKCM dataset, showcasing the methodology and providing insights into the results obtained.

### Core scripts
- **PRUNING.py** — CC-Pruning utilities and SynthTree construction.
- **DECISION_TREE_CLASSIFIER.py** — L-trim Pruning utilities and SynthTree construction.
- **co_supervision_test.py** — Main experimentation for co-supervision on explainable methods like CART and LRF. Loads data, selects/tunes a teacher, clusters & augments the training set, then trains **students** on original vs augmented data and reports metrics/interpretability with CSV summaries.
- **example_preprocessing.py** — Reusable dataset loaders & preprocessing, including min‑max/standardization, categorical encoding, consistent splits.
- **generalized_mlm.py** — Main logic for MLM-EPIC construction.

### Data
A folder named **"data"** is included in this repository, containing all datasets referenced in the related paper. This allows users to easily access and utilize the data for their own analyses and experiments.


### Misc
- **requirements.txt** — Python dependencies for the project.

---

## Installation

### Python (recommended: virtual environment)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### R (required for LRT/LRF via rpy2)
`co_supervision_test.py` uses **rpy2** to call **Rforestry**. The script loads Rforestry from a local source checkout one level **above** this repo:

```
<parent>/
├─ Rforestry/         # R package source (so devtools::load_all("../Rforestry") works)
└─ this_repo/
```

Install R dependencies in an R session:
```r
install.packages("devtools")          # if needed
# The Python script will run:
#   library(devtools); devtools::load_all("../Rforestry")
# Ensure ../Rforestry exists and is a valid R package checkout.
```

> If you don’t need the **LRT/LRF** models, you can still run experiments using the pure‑Python teachers and the **CART** student; R is only needed for LRT/LRF.

---

## Data layout

Place the following files under a `./data/` folder (paths are hard‑coded in `example_preprocessing.py`):

| Dataset label (use in CLI) | Expected file(s) / source |
|---|---|
| **SKCM** | `./data/TCGA_skcm.csv` |
| **Road Safety** | `./data/road_safety_dataset.csv` |
| **Compas** | `./data/compass_dataset.csv` |
| **Upselling** | `./data/KDDCup09_upselling_dataset.csv` |
| **Bike Sharing** | `./data/hour.csv.gz` (gzip) |
| **Abalone** | `./data/abalone.csv` |
| **Servo** | `./data/servo.csv` |
| **Cal Housing** | fetched via `sklearn.datasets.fetch_california_housing()` |

---

## Running experiments (`co_supervision_test.py`)

### CLI
```bash
python co_supervision_test.py [--runs N] [DATASET] [TEACHER]
```

**Arguments**
- `--runs, -r` *(int, default=5)* — number of repeated train/test splits for averaging.
- `DATASET` *(optional)* — one of: `SKCM`, `Road Safety`, `Compas`, `Upselling`, `Cal Housing`, `Bike Sharing`, `Abalone`, `Servo`.
- `TEACHER` *(optional)* — one of: `RF`, `GB`, `MLP`, `LRF`.

**Behavior when omitted**
- If you omit `DATASET`, the script runs **all datasets**.
- If you omit `TEACHER`, the script runs **all teachers**.

### Examples
Run **SKCM** with **RF** teacher for **10 runs**:
```bash
python co_supervision_test.py --runs 10 SKCM RF
```

Run **everything** (all datasets × all teachers) for 5 runs:
```bash
python co_supervision_test.py
```
---

## File overview (quick reference)

- `BikeSharing.ipynb` — Notebook walk‑through on Bike Sharing.  
- `SKCM.ipynb` — Notebook walk‑through on TCGA SKCM.  
- `DECISION_TREE_CLASSIFIER.py` — Main utils for L-trim Pruning  
- `PRUNING.py` — Main utils for CC-Pruning
- `co_supervision_test.py` — Main experimentation for co-supervision on explainable methods.  
- `example_preprocessing.py` — Dataset loaders & preprocessing.  
- `generalized_mlm.py` — MLM_EPIC utilities.
- `requirements.txt` — Python dependencies.

---

