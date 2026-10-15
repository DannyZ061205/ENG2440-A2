"""
Full-volume prediction with an overlapping sliding window, and connected-component postprocessing.

Sliding window: patches of size P are placed with stride S = round(P * (1 - overlap)) along each axis; the
last patch on an axis is shifted back to end exactly at the volume border, so every voxel is covered.
Each patch's sigmoid probabilities are multiplied by a blending weight and summed into a probability map;
a weight map accumulates the weights; probability = sum / weight. Only then is the 0.5 threshold applied,
so overlapping patches vote with their probabilities, not with their hard masks.

Blending weights
  gaussian  a 3D Gaussian centred on the patch (sigma = P / 8, as in nnU-Net): voxels near a patch border,
            where the network sees the least context, count less than voxels near its centre;
  constant  every voxel of every patch counts equally (plain averaging).
"""
import time

import numpy as np
import torch
from scipy import ndimage


def window_starts(dim, P, stride):
    """Start indices along one axis: 0, stride, 2*stride, ... and a last window ending at `dim`."""
    if dim <= P:
        return [0]
    starts = list(range(0, dim - P + 1, stride))
    if starts[-1] != dim - P:
        starts.append(dim - P)
    return starts


def blend_weight(P, mode="gaussian", sigma_scale=1 / 8):
    if mode == "constant":
        return np.ones((P, P, P), dtype=np.float32)
    r = np.arange(P) - (P - 1) / 2
    g = np.exp(-0.5 * (r / (sigma_scale * P)) ** 2)
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    w = w / w.max()
    return np.maximum(w, 1e-3).astype(np.float32)          # no exact zeros at the corners


@torch.no_grad()
def predict_volume(model, image, patch_size, overlap=0.5, blend="gaussian", batch_size=4, device="cpu",
                   return_weight=False):
    """Probability map [X, Y, Z] (float32) for a channel-first image [C, X, Y, Z] (numpy, any float dtype)."""
    P = patch_size
    stride = max(1, int(round(P * (1 - overlap))))
    C, *shape = image.shape
    padded = [max(s, P) for s in shape]
    if padded != list(shape):
        image = np.pad(image, [(0, 0)] + [(0, p - s) for p, s in zip(padded, shape)])
    corners = [(x, y, z) for x in window_starts(padded[0], P, stride)
               for y in window_starts(padded[1], P, stride)
               for z in window_starts(padded[2], P, stride)]
    w = blend_weight(P, blend)
    w_t = torch.from_numpy(w).to(device)
    prob_sum = torch.zeros(padded, dtype=torch.float32, device=device)
    weight_sum = torch.zeros(padded, dtype=torch.float32, device=device)
    image_t = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).to(device)
    model.eval()
    for i in range(0, len(corners), batch_size):
        batch = corners[i:i + batch_size]
        x = torch.stack([image_t[:, a:a + P, b:b + P, c:c + P] for a, b, c in batch])
        p = torch.sigmoid(model(x))[:, 0]
        for (a, b, c), pk in zip(batch, p):
            prob_sum[a:a + P, b:b + P, c:c + P] += pk * w_t
            weight_sum[a:a + P, b:b + P, c:c + P] += w_t
    prob = (prob_sum / weight_sum).cpu().numpy()[:shape[0], :shape[1], :shape[2]]
    if return_weight:
        return prob, weight_sum.cpu().numpy()[:shape[0], :shape[1], :shape[2]], corners
    return prob


def timed_predict(*args, **kwargs):
    """predict_volume plus its wall-clock run time in seconds (device synchronised)."""
    t = time.perf_counter()
    out = predict_volume(*args, **kwargs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return out, time.perf_counter() - t


# ---------------------------------------------------------- postprocessing --
CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)               # 26-connectivity: faces, edges and corners


def components(mask):
    """Label the connected components of a binary mask: (labels array, sizes in voxels per component)."""
    labels, n = ndimage.label(mask, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    return labels, sizes


def remove_small_components(mask, min_voxels):
    """Keep only connected components with at least `min_voxels` voxels."""
    if min_voxels <= 0 or not mask.any():
        return mask.astype(np.uint8)
    labels, sizes = components(mask)
    keep = np.concatenate([[False], sizes >= min_voxels])
    return keep[labels].astype(np.uint8)


def keep_largest_component(mask):
    """Keep only the largest connected component (empty masks stay empty)."""
    if not mask.any():
        return mask.astype(np.uint8)
    labels, sizes = components(mask)
    return (labels == (np.argmax(sizes) + 1)).astype(np.uint8)
