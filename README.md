# ENG2440 Assignment 2: Whole-Tumour Segmentation in Multimodal 3D Brain MRI

Zichen Zhao · 25011189 · ENG 2440 Medical Imaging & AI in Healthcare, Fall 2026 ·
<https://github.com/DannyZ061205/ENG2440-A2>

An end-to-end PyTorch pipeline that segments the whole brain tumour (oedema + non-enhancing + enhancing tumour)
from four co-registered MRI channels (FLAIR, T1, T1ce, T2) of the Medical Segmentation Decathlon Task01
BrainTumour data. It inspects NIfTI geometry, preprocesses volumes without corrupting spatial information,
trains a lightweight 3D U-Net on 64³ patches, compares uniform with tumour-aware patch sampling under
controlled conditions, reconstructs full volumes with an overlapping sliding window, and evaluates overlap,
boundary (HD95, mm) and volume (mL) errors with a three-plane error analysis. Two stress tests (missing
modality, sliding-window seams) are included. This is a learning exercise, not a clinical tool.

**This repository contains no NIfTI files, labels or split tables.** The executed notebook displays a few
MRI slices, as the assignment requires. The data must be downloaded separately (see below).

## Files

| file | what it is |
|---|---|
| `25011189_ENG2440_A2.ipynb` | the fully executed notebook (all code, outputs, figures and written answers) |
| `helpers/volumes.py` | NIfTI reading, geometry facts, RAS+ reorientation, brain crop, per-channel z-score, preprocessing cache |
| `helpers/patches.py` | patch `Dataset` with the uniform and tumour-aware samplers and the training-only augmentation |
| `helpers/unet3d.py` | the lightweight 3D U-Net |
| `helpers/train.py` | loss, one controlled training run, whole-volume validation tracking, run caching |
| `helpers/inference.py` | overlapping sliding-window inference with Gaussian blending; connected-component postprocessing |
| `helpers/metrics.py` | Dice, IoU, sensitivity, HD95 (mm), volumes (mL), bootstrap confidence intervals |
| `helpers/evaluate.py` | whole-split prediction and scoring in the full image grid |
| `helpers/plotting.py` | orientation-correct plane display, fixed fields of view, contours, TP/FP/FN error maps, collision-free point labels |
| `helpers/checklist.py` | offline audit: every requirement has evidence in the executed notebook; every number in the report appears in a notebook output |
| `report/main.tex`, `report/make_tables.py`, `report/tables/` | LaTeX source of the report and the script that writes its tables from the notebook's saved tables |
| `helpers/paper_plot_style.py` | figure house style and automatic figure audit, copied from Assignment 1 (source: the *paper-figure* figure-style skill, Appendix A) |
| `figures/methods.json` | one label / colour / line style per sampler, used by every figure |
| `environment.yml`, `requirements.txt` | the environment (exact package versions used) |

## Reproduce

1. Download `Task01_BrainTumour.tar` (7.6 GB) from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
   or the MONAI mirror `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar`, and extract it next
   to the notebook so that `Task01_BrainTumour/imagesTr`, `labelsTr` and `dataset.json` exist. Only the 484 labelled
   cases are used. Put `assignment2_split.csv` (from Moodle) next to the notebook.
2. `conda env create -f environment.yml && conda activate eng2440-a2`
   (or `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`).
3. Run the notebook top to bottom. The first run builds a 13 GB preprocessing cache in `data/cache/` (about 30 s with
   8 processes) and trains six runs (two samplers × seeds 0, 1, 2) of 3,000 iterations each, 52–64 minutes per run on an
   Apple M4 Max GPU (MPS), about 5.5 hours in total; a recent CUDA GPU is faster. Trained runs are stored in
   `outputs/runs/` with their configuration, and a saved run is reused only if its configuration matches `CONFIG` exactly
   (set `"retrain": True` to force retraining). With the runs in place, executing the whole notebook (evaluation, error
   analysis and both stress tests) takes about 27 minutes.
4. Check that every requirement is covered and that the report's numbers match the notebook outputs:
   `python -m helpers.checklist 25011189_ENG2440_A2.ipynb report/main.tex`.
5. Report: `python report/make_tables.py` writes the LaTeX tables from `outputs/tables/` (saved by the notebook), then
   `cd report && tectonic main.tex`.

Note: MPS kernels are not guaranteed to be bit-for-bit deterministic, so retraining can differ from the saved runs in
the last decimals.

## Data attribution

Medical Segmentation Decathlon, Task01 BrainTumour (CC-BY-SA 4.0): Antonelli et al., *Nature Communications* 2022;
Simpson et al., arXiv:1902.09063 (2019). The cases derive from the BraTS 2016/2017 challenges (Menze et al., *IEEE
TMI* 2015; Bakas et al., *Scientific Data* 2017).
