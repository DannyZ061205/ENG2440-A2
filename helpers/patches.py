"""
Patch sampling and training-only augmentation.

`PatchDataset` is a map-style PyTorch Dataset whose item i is a random patch drawn with its own random
generator, seeded by (seed, i). The sequence of patches is therefore fixed by the seed alone: it does not
depend on the number of DataLoader workers or on their scheduling, and two runs that differ only in the
sampler see the same case order and the same augmentation draws wherever the samplers agree.

Samplers
  uniform   the patch corner is uniform over all positions where the patch fits inside the cropped volume;
  tumour    with probability p_tumour the patch is centred on a randomly chosen tumour voxel (clipped so
            the patch stays inside the volume), otherwise it is drawn uniformly as above.

Augmentation (training only; each transform imitates a real source of variation)
  left-right flip      p = 0.5. Gliomas occur in either hemisphere and the brain is close to mirror
                       symmetric, so a mirrored brain is still a plausible brain. No anterior-posterior or
                       superior-inferior flips: those create anatomy that does not exist.
  axial rotation       p = 0.5, angle ~ U(-10, 10) degrees about the S axis: head tilt in the scanner that
                       co-registration did not fully remove. Linear interpolation for MRI, nearest
                       neighbour for the label, so label values stay in {0, 1, 2, 3}.
  intensity scale/shift  per channel, x U(0.9, 1.1) and + U(-0.1, 0.1) SD on brain voxels only: residual
                       scanner gain / contrast differences left after z-scoring. Background stays 0.
"""
import math

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from helpers.volumes import load_cached


def extract(array, start, size):
    """array[..., start:start+size] over the last three axes; out-of-range parts are zero-filled."""
    out = np.zeros(array.shape[:-3] + tuple(size), dtype=array.dtype)
    src, dst = [], []
    for s, n, dim in zip(start, size, array.shape[-3:]):
        a, b = max(s, 0), min(s + n, dim)
        src.append(slice(a, b))
        dst.append(slice(a - s, b - s))
    out[(Ellipsis,) + tuple(dst)] = array[(Ellipsis,) + tuple(src)]
    return out


class PatchDataset(Dataset):
    """Random training patches: item i -> (image [C, P, P, P] float32, mask [1, P, P, P] float32)."""

    def __init__(self, case_ids, cache_dir, patch_size, n_samples, sampler="uniform", p_tumour=0.5,
                 seed=0, augment=True, rotate_deg=10.0):
        assert sampler in ("uniform", "tumour")
        self.case_ids = list(case_ids)
        self.cache_dir = cache_dir
        self.P = int(patch_size)
        self.n_samples = int(n_samples)
        self.sampler = sampler
        self.p_tumour = float(p_tumour) if sampler == "tumour" else 0.0
        self.seed = int(seed)
        self.augment = augment
        self.rotate_deg = float(rotate_deg)
        self._fg = {}

    def __len__(self):
        return self.n_samples

    def _tumour_coords(self, case_id):
        if case_id not in self._fg:
            self._fg[case_id] = np.load(f"{self.cache_dir}/{case_id}_fg.npy")
        return self._fg[case_id]

    def draw(self, i):
        """The sampling decision for item i: (case_id, patch start [x, y, z], aimed_at_tumour, rng)."""
        rng = np.random.default_rng([self.seed, i])
        case_id = self.case_ids[rng.integers(len(self.case_ids))]
        image, _, meta = load_cached(case_id, self.cache_dir, with_label=False)
        shape = image.shape[1:]
        # every random number is drawn whether or not it is used, so both samplers consume the same stream
        aim = rng.random() < self.p_tumour
        uniform_start = [int(rng.integers(0, d - self.P + 1)) for d in shape]
        u = rng.random()
        if aim:
            fg = self._tumour_coords(case_id)
            centre = fg[int(u * len(fg))]
            start = [int(np.clip(c - self.P // 2, 0, d - self.P)) for c, d in zip(centre, shape)]
        else:
            start = uniform_start
        return case_id, start, aim, rng

    def __getitem__(self, i):
        case_id, start, aim, rng = self.draw(i)
        image, label, _ = load_cached(case_id, self.cache_dir)
        flip, rotate = rng.random() < 0.5, rng.random() < 0.5
        angle = rng.uniform(-self.rotate_deg, self.rotate_deg)
        scale = rng.uniform(0.9, 1.1, size=image.shape[0])
        shift = rng.uniform(-0.1, 0.1, size=image.shape[0])

        if self.augment and rotate:
            # cut a larger in-plane region so the rotated patch has no empty corners, rotate, crop the centre
            m = math.ceil(self.P * (abs(math.cos(math.radians(angle))) + abs(math.sin(math.radians(angle)))))
            m += (m - self.P) % 2
            off = (m - self.P) // 2
            big = (start[0] - off, start[1] - off, start[2])
            img = extract(image, big, (m, m, self.P)).astype(np.float32)
            lab = extract(label, big, (m, m, self.P))
            img = ndimage.rotate(img, angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
            lab = ndimage.rotate(lab, angle, axes=(0, 1), reshape=False, order=0, mode="constant", cval=0)
            img = img[:, off:off + self.P, off:off + self.P]
            lab = lab[off:off + self.P, off:off + self.P]
        else:
            img = extract(image, start, (self.P,) * 3).astype(np.float32)
            lab = extract(label, start, (self.P,) * 3)

        if self.augment:
            if flip:                                   # axis 0 of [X, Y, Z] is left-right in RAS+
                img, lab = img[:, ::-1], lab[::-1]
            brain = img != 0
            img = np.where(brain, img * scale[:, None, None, None] + shift[:, None, None, None], 0.0)

        mask = (lab > 0).astype(np.float32)[None]
        return {"image": torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
                "mask": torch.from_numpy(np.ascontiguousarray(mask)),
                "case_id": case_id, "aimed_at_tumour": aim, "has_tumour": bool(mask.any())}
