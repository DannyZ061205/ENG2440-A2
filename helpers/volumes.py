"""
Reading the MSD Task01 NIfTI volumes, checking their geometry, and preprocessing them into a cache.

Every array returned here is in RAS+ voxel order (nibabel's closest canonical orientation): increasing
array index 0 moves towards the patient's Right (R), index 1 towards Anterior (A), index 2 towards
Superior (S). Images are channel-first: [C, X, Y, Z] with the
channel order of dataset.json (FLAIR, T1, T1ce, T2). In PyTorch terms [C, D, H, W] = [C, X, Y, Z].

Preprocessing of one case (`preprocess_case`):
  1. load image + label, reorient both to RAS+ (a pure axis permutation/flip: no interpolation);
  2. crop both to the bounding box of the brain (voxels that are non-zero in any channel);
  3. pad at the far end of each axis up to at least the patch size (zeros = background);
  4. z-score each channel with the mean / SD of that channel's non-zero (brain) voxels, keep the
     background at exactly 0;
  5. save image (float16), binary-tumour coordinates for patch sampling and, for train / validation only,
     the original multi-class label (uint8), plus a JSON file with the crop box and statistics.
Test labels are never cached: they are read from the NIfTI files only in the final evaluation.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

CHANNELS = ["FLAIR", "T1", "T1ce", "T2"]          # order of the 4th NIfTI axis (dataset.json "modality")
LABEL_NAMES = {0: "background", 1: "oedema", 2: "non-enhancing tumour", 3: "enhancing tumour"}
CACHE_VERSION = 1                                  # bump when the preprocessing changes


# ------------------------------------------------------------------ reading --
def header_facts(image_path, label_path):
    """Geometry facts read from the NIfTI headers only (no voxel data is loaded)."""
    im, lb = nib.load(str(image_path)), nib.load(str(label_path))
    return {
        "image_shape": tuple(int(s) for s in im.shape),
        "label_shape": tuple(int(s) for s in lb.shape),
        "image_dtype": str(im.get_data_dtype()),
        "label_dtype": str(lb.get_data_dtype()),
        "orientation": "".join(nib.aff2axcodes(im.affine)),
        "label_orientation": "".join(nib.aff2axcodes(lb.affine)),
        "spacing_mm": tuple(float(z) for z in im.header.get_zooms()[:3]),
        "label_spacing_mm": tuple(float(z) for z in lb.header.get_zooms()[:3]),
        "same_affine": bool(np.allclose(im.affine, lb.affine)),
        "affine": im.affine,
    }


def load_image(image_path):
    """4-channel MRI as float32 [C, X, Y, Z] in RAS+ order, plus the RAS+ affine and voxel spacing (mm)."""
    im = nib.as_closest_canonical(nib.load(str(image_path)))
    data = np.asarray(im.dataobj, dtype=np.float32)            # [X, Y, Z, C]
    return np.ascontiguousarray(np.moveaxis(data, -1, 0)), im.affine, tuple(float(z) for z in im.header.get_zooms()[:3])


def load_label(label_path):
    """Original multi-class label (uint8, values 0-3) [X, Y, Z] in RAS+ order."""
    lb = nib.as_closest_canonical(nib.load(str(label_path)))
    return np.asarray(lb.dataobj).astype(np.uint8)


def whole_tumour(label):
    """Binary whole tumour: 1 wherever the reference has any tumour sub-region label (1, 2 or 3)."""
    return (label > 0).astype(np.uint8)


# ------------------------------------------------------------ preprocessing --
def brain_box(image):
    """Bounding box (list of (start, stop) per spatial axis) of voxels that are non-zero in any channel."""
    mask = np.any(image != 0, axis=0)
    box = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        idx = np.flatnonzero(mask.any(axis=other))
        box.append((int(idx[0]), int(idx[-1]) + 1))
    return box


def zscore_brain(image):
    """Per-channel z-score using the mean / SD of that channel's non-zero voxels; background stays 0."""
    out = np.zeros_like(image, dtype=np.float32)
    stats = []
    for c in range(image.shape[0]):
        brain = image[c] != 0
        values = image[c][brain]
        mean, sd = float(values.mean()), float(values.std())
        out[c][brain] = (values - mean) / sd
        stats.append({"raw_mean": mean, "raw_sd": sd,
                      "norm_mean": float(out[c][brain].mean()), "norm_sd": float(out[c][brain].std())})
    return out, stats


def pad_to(array, min_size, value=0):
    """Pad the last three axes at their far end so each is at least `min_size` long."""
    pads = [(0, 0)] * (array.ndim - 3) + [(0, max(0, min_size - s)) for s in array.shape[-3:]]
    return np.pad(array, pads, constant_values=value)


def preprocess_case(case_id, image_path, label_path, cache_dir, min_size, cache_label, n_fg_coords=4000):
    """Crop, pad, normalise one case and write it to `cache_dir`. Returns the metadata dict."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / f"{case_id}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("version") == CACHE_VERSION and meta.get("min_size") == min_size \
                and meta.get("label_cached") == cache_label:
            return meta

    image, affine, spacing = load_image(image_path)
    box = brain_box(image)
    crop = tuple(slice(a, b) for a, b in box)
    image = pad_to(image[(slice(None),) + crop], min_size)
    image, stats = zscore_brain(image)
    np.save(cache_dir / f"{case_id}_img.npy", image.astype(np.float16))

    meta = {"version": CACHE_VERSION, "case_id": case_id, "min_size": min_size, "label_cached": cache_label,
            "full_shape": [int(s) for s in nib.load(str(image_path)).shape[:3]], "box": box,
            "cropped_shape": [int(b - a) for a, b in box], "cache_shape": list(image.shape[1:]),
            "spacing_mm": list(spacing), "affine": np.asarray(affine).tolist(), "channel_stats": stats,
            "brain_voxels": int(np.any(image != 0, axis=0).sum())}

    if cache_label:
        label = load_label(label_path)
        tumour = whole_tumour(label)
        # every tumour voxel must lie inside the brain box, or cropping would lose reference voxels
        assert tumour.sum() == tumour[crop].sum(), f"{case_id}: tumour voxels outside the brain box"
        label = pad_to(label[crop], min_size)
        np.save(cache_dir / f"{case_id}_lab.npy", label)
        fg = np.argwhere(label > 0).astype(np.int16)
        rng = np.random.default_rng(0)
        if len(fg) > n_fg_coords:                               # a random subset is enough for sampling centres
            fg = fg[rng.choice(len(fg), n_fg_coords, replace=False)]
        np.save(cache_dir / f"{case_id}_fg.npy", fg)
        meta["tumour_voxels"] = int((label > 0).sum())
        meta["label_values"] = [int(v) for v in np.unique(label)]
    meta_path.write_text(json.dumps(meta))
    return meta


def load_cached(case_id, cache_dir, with_label=True, mmap=True):
    """Cached (image float16 [C,X,Y,Z], multi-class label uint8 [X,Y,Z] or None, meta dict)."""
    cache_dir = Path(cache_dir)
    mode = "r" if mmap else None
    image = np.load(cache_dir / f"{case_id}_img.npy", mmap_mode=mode)
    meta = json.loads((cache_dir / f"{case_id}.json").read_text())
    label = np.load(cache_dir / f"{case_id}_lab.npy", mmap_mode=mode) if with_label and meta["label_cached"] else None
    return image, label, meta


def uncrop(cropped, meta):
    """Place a cache-grid array [X', Y', Z'] back into the full RAS+ grid (zeros outside the brain box)."""
    full = np.zeros(meta["full_shape"], dtype=cropped.dtype)
    (x0, x1), (y0, y1), (z0, z1) = meta["box"]
    full[x0:x1, y0:y1, z0:z1] = cropped[:x1 - x0, :y1 - y0, :z1 - z0]   # drop the padding
    return full


# -------------------------------------------------------------- resampling --
def resample(volume, spacing, new_spacing, order):
    """Resample a 3D array from `spacing` to `new_spacing` (mm). order=1: linear (MRI), order=0: nearest
    neighbour (labels: no new, in-between label values can appear)."""
    factors = [s / n for s, n in zip(spacing, new_spacing)]
    return ndimage.zoom(volume, factors, order=order, mode="nearest", grid_mode=True)


def _preprocess_job(args):
    return preprocess_case(*args)


def preprocess_all(split_df, data_root, cache_dir, min_size, workers=8):
    """Preprocess every case of the split table in parallel (cached cases are skipped).
    Test labels are not cached (cache_label=False for the test split)."""
    from concurrent.futures import ProcessPoolExecutor
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(r.case_id, Path(data_root) / r.image_relpath, Path(data_root) / r.label_relpath, str(cache_dir),
             min_size, r.split != "test") for r in split_df.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        metas = list(ex.map(_preprocess_job, jobs, chunksize=4))
    return {m["case_id"]: m for m in metas}
