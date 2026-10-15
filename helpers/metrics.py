"""
Case-level segmentation metrics in physical units, and bootstrap confidence intervals.

All metrics compare a binary prediction P with the binary reference R of one case (whole tumour).
  Dice         2|P∩R| / (|P| + |R|)
  IoU          |P∩R| / |P∪R|
  sensitivity  |P∩R| / |R|                      (fraction of reference tumour found)
  FP volume    |P \\ R| x voxel volume, in mL  (tumour predicted where the reference has none)
  HD95         95th-percentile symmetric surface distance, in mm. Surfaces are the voxels of each mask
               that touch the background (mask minus its 26-connected erosion). For every surface voxel of
               P the Euclidean distance (scaled by the voxel spacing) to the nearest surface voxel of R is
               found, and vice versa; HD95 = max(95th percentile of P->R, 95th percentile of R->P), the
               definition used by the BraTS evaluation and MONAI. Undefined (NaN) if P or R is empty.
  volume       voxel count x product of voxel spacing (mm^3) / 1000 = mL; volume error = predicted - reference.
"""
import numpy as np
from scipy import ndimage

_STRUCT = np.ones((3, 3, 3), dtype=bool)


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=_STRUCT, border_value=0)


def hd95(pred, ref, spacing):
    """95th-percentile Hausdorff distance (mm) between two binary masks; NaN if either is empty."""
    pred, ref = pred.astype(bool), ref.astype(bool)
    if not pred.any() or not ref.any():
        return float("nan")
    # work inside the joint bounding box (plus a margin) to keep the distance transforms small
    both = pred | ref
    box = tuple(slice(max(int(i.min()) - 2, 0), int(i.max()) + 3) for i in np.nonzero(both))
    pred, ref = pred[box], ref[box]
    sp, sr = _surface(pred), _surface(ref)
    dist_to_ref = ndimage.distance_transform_edt(~sr, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return float(max(np.percentile(dist_to_ref[sp], 95), np.percentile(dist_to_pred[sr], 95)))


def case_metrics(pred, ref, spacing, with_hd95=True):
    """Dictionary of overlap, boundary and volume metrics for one case."""
    pred, ref = pred.astype(bool), ref.astype(bool)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    n_pred, n_ref = tp + fp, tp + fn
    out = {
        "dice": 2 * tp / (n_pred + n_ref) if n_pred + n_ref else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "sensitivity": tp / n_ref if n_ref else float("nan"),
        "precision": tp / n_pred if n_pred else float("nan"),
        "fp_ml": fp * voxel_ml,
        "fn_ml": fn * voxel_ml,
        "ref_ml": n_ref * voxel_ml,
        "pred_ml": n_pred * voxel_ml,
        "empty_pred": n_pred == 0,
    }
    out["vol_err_ml"] = out["pred_ml"] - out["ref_ml"]
    out["abs_vol_err_ml"] = abs(out["vol_err_ml"])
    out["vol_err_pct"] = 100 * out["vol_err_ml"] / out["ref_ml"] if n_ref else float("nan")
    out["abs_vol_err_pct"] = abs(out["vol_err_pct"])
    if with_hd95:
        out["hd95_mm"] = hd95(pred, ref, spacing)
    return out


def bootstrap_ci(values, n_boot=2000, seed=0, stat=np.nanmean, level=0.95):
    """Percentile bootstrap CI over cases for a statistic of `values` (NaNs ignored by the default)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = np.array([stat(values[i]) for i in idx])
    a = (1 - level) / 2
    return float(np.nanquantile(stats, a)), float(np.nanquantile(stats, 1 - a))


def paired_bootstrap_diff(a, b, n_boot=2000, seed=0, level=0.95):
    """Mean of (a - b) over paired cases and its percentile bootstrap CI."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed, level=level)
    return float(np.nanmean(d)), lo, hi
