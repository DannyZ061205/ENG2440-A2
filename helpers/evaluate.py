"""
Whole-volume evaluation used by sections E, F, G and the bonus tests.

`predict_split` runs the sliding window over every case of a split and returns hard masks in the cache grid
(the cropped brain box); `score_split` puts each mask back into the full 240 x 240 x 155 RAS+ grid, optionally
applies the connected-component rule, and compares it with the reference read from the original label NIfTI.
References are loaded explicitly by the caller (`load_references`), which is how the test labels stay unread
until the final evaluation.
"""
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from helpers.inference import predict_volume, remove_small_components
from helpers.metrics import case_metrics
from helpers.volumes import load_cached, uncrop, whole_tumour


def load_references(case_ids, split_df, data_root):
    """Binary whole-tumour reference in the full RAS+ grid and voxel spacing (mm), per case."""
    by_id = split_df.set_index("case_id")
    refs = {}
    for cid in case_ids:
        lab = nib.as_closest_canonical(nib.load(f"{data_root}/{by_id.loc[cid, 'label_relpath']}"))
        refs[cid] = (whole_tumour(np.asarray(lab.dataobj)), tuple(float(z) for z in lab.header.get_zooms()[:3]))
    return refs


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def predict_split(model, case_ids, cfg, device, overlap=None, blend=None, zero_channel=None, keep_prob=()):
    """Hard masks (uint8, cache grid) per case, seconds per case, and probability maps for `keep_prob` cases."""
    overlap = cfg["sw_overlap"] if overlap is None else overlap
    blend = cfg["sw_blend"] if blend is None else blend
    masks, seconds, probs = {}, {}, {}
    for cid in case_ids:
        image = np.asarray(load_cached(cid, cfg["cache_dir"], with_label=False)[0], dtype=np.float32)
        if zero_channel is not None:
            image = image.copy()
            image[zero_channel] = 0.0                  # = the channel's brain mean after z-scoring
        _sync()
        t = time.perf_counter()
        prob = predict_volume(model, image, cfg["patch_size"], overlap=overlap, blend=blend,
                              batch_size=cfg["sw_batch"], device=device)
        _sync()
        seconds[cid] = time.perf_counter() - t
        masks[cid] = (prob >= cfg["threshold"]).astype(np.uint8)
        if cid in keep_prob:
            probs[cid] = prob
    return masks, seconds, probs


def full_mask(mask, case_id, cfg, min_voxels=0):
    """Cache-grid mask -> full-grid mask, then remove components smaller than `min_voxels`."""
    meta = load_cached(case_id, cfg["cache_dir"], with_label=False)[2]
    return remove_small_components(uncrop(mask, meta), min_voxels)


def score_split(masks, refs, cfg, min_voxels=0, with_hd95=True):
    """Per-case metrics (DataFrame indexed by case_id) of full-grid predictions against the references."""
    rows = []
    for cid, mask in masks.items():
        ref, spacing = refs[cid]
        rows.append({"case_id": cid, **case_metrics(full_mask(mask, cid, cfg, min_voxels), ref, spacing,
                                                    with_hd95=with_hd95)})
    return pd.DataFrame(rows).set_index("case_id")


def summarise(df, metrics=("dice", "iou", "sensitivity", "hd95_mm", "fp_ml", "abs_vol_err_ml", "abs_vol_err_pct")):
    """Mean, SD, median over cases (NaN HD95 of empty predictions excluded) and the number of empty predictions."""
    out = pd.DataFrame({m: [df[m].mean(), df[m].std(), df[m].median()] for m in metrics if m in df},
                       index=["mean", "SD", "median"])
    out.loc["empty predictions"] = int(df["empty_pred"].sum())
    return out
