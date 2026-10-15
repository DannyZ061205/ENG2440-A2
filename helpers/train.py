"""
Training one patch-based run and tracking it on whole validation volumes.

A run is defined by the shared settings in CONFIG plus (sampler, seed). Two runs with the same seed start
from the *same* initial weights (saved once per seed in outputs/runs/init_seed{seed}.pt), use the same
optimiser, learning-rate schedule, loss, iteration budget and patch stream seed; only the sampler differs.

Loss: soft Dice + binary cross-entropy, both on the whole batch. The batch-level soft Dice
1 - 2 sum(p*y) / (sum(p) + sum(y)) is defined even for patches without tumour; BCE gives every voxel a
gradient, Dice counters the class imbalance (tumour is a few percent of brain voxels).

Monitoring: every `eval_every` iterations the model predicts every validation case and a fixed subset of
training cases with the full sliding-window pipeline. For each case the same Dice + BCE loss is computed on
the whole (cropped) volume, plus the Dice of the thresholded mask. Because training and validation loss
are computed the same way on whole volumes, their gap measures overfitting and is comparable between the
two samplers (the running patch loss is not: it depends on what the sampler shows the network).
The checkpoint with the highest mean validation Dice is kept ("best"); the test split is never touched.

A saved run is reused only if its stored configuration equals the requested one exactly.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from helpers.inference import predict_volume
from helpers.patches import PatchDataset
from helpers.unet3d import UNet3D
from helpers.volumes import load_cached

RUN_KEYS = ["patch_size", "batch_size", "iterations", "eval_every", "widths", "lr", "weight_decay",
            "warmup_iters", "p_tumour", "rotate_deg", "sw_overlap", "sw_blend", "threshold",
            "n_monitor_train_cases"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_model(cfg):
    return UNet3D(in_channels=4, out_channels=1, widths=tuple(cfg["widths"]))


def dice_bce_loss(logits, target, eps=1.0):
    """Soft Dice (over the whole batch) + BCE, from logits."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = (2 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)
    return bce + (1 - dice)


def volume_loss(prob, target, eps=1.0):
    """The same Dice + BCE loss evaluated on a whole-volume probability map (numpy)."""
    p = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    y = target.astype(np.float64)
    bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    dice = (2 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)
    return float(bce + 1 - dice)


def evaluate_cases(model, case_ids, cfg, device):
    """Whole-volume loss and Dice (threshold 0.5) per cached case."""
    rows = []
    for cid in case_ids:
        image, label, _ = load_cached(cid, cfg["cache_dir"])
        target = (np.asarray(label) > 0)
        prob = predict_volume(model, image, cfg["patch_size"], overlap=cfg["sw_overlap"],
                              blend=cfg["sw_blend"], batch_size=cfg["sw_batch"], device=device)
        pred = prob >= cfg["threshold"]
        inter, total = (pred & target).sum(), pred.sum() + target.sum()
        rows.append({"case_id": cid, "loss": volume_loss(prob, target),
                     "dice": float(2 * inter / total) if total else 1.0})
    return pd.DataFrame(rows)


def lr_factor(it, cfg):
    """Linear warm-up, then cosine decay to zero at the end of the budget."""
    if it < cfg["warmup_iters"]:
        return (it + 1) / cfg["warmup_iters"]
    t = (it - cfg["warmup_iters"]) / max(1, cfg["iterations"] - cfg["warmup_iters"])
    return 0.5 * (1 + math.cos(math.pi * t))


def run_config(cfg, sampler, seed, train_ids, val_ids, monitor_ids):
    rc = {k: cfg[k] for k in RUN_KEYS}
    rc["widths"] = list(rc["widths"])
    rc.update(sampler=sampler, seed=seed, train_ids=list(train_ids), val_ids=list(val_ids),
              monitor_ids=list(monitor_ids))
    return rc


def initial_weights(cfg, seed, device):
    """The seed's initial weights, created once and shared by every sampler trained with that seed."""
    path = Path(cfg["run_dir"]) / f"init_seed{seed}_{'-'.join(map(str, cfg['widths']))}.pt"
    set_seed(seed)
    model = build_model(cfg)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model.to(device), path


def train_run(cfg, sampler, seed, train_ids, val_ids, monitor_ids, device, retrain=False, log=print):
    """Train (or reload) one run. Returns the run directory."""
    rc = run_config(cfg, sampler, seed, train_ids, val_ids, monitor_ids)
    out = Path(cfg["run_dir"]) / f"{sampler}_seed{seed}"
    done = out / "summary.json"
    if not retrain and done.exists() and json.loads((out / "config.json").read_text()) == rc:
        log(f"{out.name}: reusing the saved run (configuration matches)")
        return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(rc, indent=1))
    done.unlink(missing_ok=True)

    model, init_path = initial_weights(cfg, seed, device)
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: lr_factor(it, cfg))
    data = PatchDataset(train_ids, cfg["cache_dir"], cfg["patch_size"], cfg["iterations"] * cfg["batch_size"],
                        sampler=sampler, p_tumour=cfg["p_tumour"], seed=seed, augment=True,
                        rotate_deg=cfg["rotate_deg"])
    loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
                        persistent_workers=cfg["num_workers"] > 0, prefetch_factor=4 if cfg["num_workers"] else None)

    iter_rows, eval_rows, best = [], [], -1.0
    t0 = time.time()
    model.train()
    for it, batch in enumerate(loader):
        x, y = batch["image"].to(device), batch["mask"].to(device)
        loss = dice_bce_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        iter_rows.append({"iteration": it + 1, "loss": loss.item(), "lr": opt.param_groups[0]["lr"],
                          "tumour_patches": int(batch["has_tumour"].sum()),
                          "aimed_patches": int(batch["aimed_at_tumour"].sum()),
                          "tumour_voxel_frac": float(y.mean())})
        if (it + 1) % cfg["eval_every"] == 0 or it + 1 == cfg["iterations"]:
            te = time.time()
            val = evaluate_cases(model, val_ids, cfg, device)
            mon = evaluate_cases(model, monitor_ids, cfg, device)
            model.train()
            recent = pd.DataFrame(iter_rows[-cfg["eval_every"]:])
            row = {"iteration": it + 1, "train_patch_loss": recent["loss"].mean(),
                   "train_loss": mon["loss"].mean(), "train_dice": mon["dice"].mean(),
                   "val_loss": val["loss"].mean(), "val_dice": val["dice"].mean(),
                   "elapsed_min": (time.time() - t0) / 60, "eval_s": time.time() - te}
            eval_rows.append(row)
            pd.DataFrame(eval_rows).to_csv(out / "history.csv", index=False)
            pd.DataFrame(iter_rows).to_csv(out / "iterations.csv", index=False)
            if row["val_dice"] > best:
                best = row["val_dice"]
                torch.save(model.state_dict(), out / "best.pt")
                val.to_csv(out / "val_cases_best.csv", index=False)
            log(f"{out.name} it {it + 1:5d} | patch loss {row['train_patch_loss']:.3f} | train loss "
                f"{row['train_loss']:.3f} dice {row['train_dice']:.3f} | val loss {row['val_loss']:.3f} "
                f"dice {row['val_dice']:.3f} | {row['elapsed_min']:.1f} min")
    torch.save(model.state_dict(), out / "last.pt")
    hist = pd.DataFrame(eval_rows)
    b = hist.loc[hist["val_dice"].idxmax()]
    summary = {"best_iteration": int(b["iteration"]), "best_val_dice": float(b["val_dice"]),
               "minutes": (time.time() - t0) / 60, "init_weights": init_path.name,
               "torch": torch.__version__, "device": str(device)}
    done.write_text(json.dumps(summary, indent=1))
    return out


def load_run(run_dir, cfg, which="best", device="cpu"):
    """(model with the chosen checkpoint, history DataFrame, per-iteration DataFrame, summary dict)."""
    run_dir = Path(run_dir)
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / f"{which}.pt", map_location="cpu"))
    model.to(device).eval()
    return (model, pd.read_csv(run_dir / "history.csv"), pd.read_csv(run_dir / "iterations.csv"),
            json.loads((run_dir / "summary.json").read_text()))


def monitor_cases(train_ids, n, seed=0):
    """A fixed random subset of training cases, evaluated like validation to measure the train/val gap."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(sorted(train_ids), n, replace=False).tolist())
