"""Fine-tune an ImageNet backbone for normal-vs-defective classification.

Training recipe
---------------
1. *Head warm-up* (``freeze_epochs``): backbone frozen, only the new 2-class head is trained, so the
   randomly initialised head does not push large gradients into the pretrained features.
2. *Full fine-tuning*: everything unfrozen, AdamW with discriminative learning rates
   (backbone LR = head LR x ``backbone_lr_mult``), linear warm-up + cosine decay.
3. Class imbalance: inverse-frequency class-weighted cross-entropy (default) or a class-balanced
   sampler; the decision threshold is tuned on validation to meet a defect-recall target.
4. Model selection / early stopping on validation loss (unweighted, so it is comparable across
   imbalance strategies). The test set is never touched here.

Usage::

    python -m defect_detection.training.train --config configs/efficientnet_b0.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from defect_detection import CLASS_NAMES
from defect_detection.data.dataset import (
    DefectDataset,
    balanced_sampler,
    inverse_frequency_weights,
    load_split,
    simulate_prevalence,
)
from defect_detection.data.transforms import EvalTransform, TrainTransform
from defect_detection.evaluation.metrics import compute_metrics, select_threshold
from defect_detection.models import build_model, param_groups, set_backbone_trainable
from defect_detection.preprocessing import PreprocessConfig

log = logging.getLogger("train")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (labels, p_defective, mean unweighted CE loss)."""
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n += len(y)
        ps.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), loss_sum / max(n, 1)


def lr_lambda_factory(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def train(cfg: dict) -> Path:
    seed_everything(cfg["seed"])
    tcfg, dcfg = cfg["train"], cfg["data"]
    device = pick_device(tcfg.get("device", "auto"))
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    file_handler = logging.FileHandler(run_dir / "train.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    log.info("Run dir %s | device %s", run_dir, device)

    pp = PreprocessConfig(image_size=int(cfg["preprocess"]["image_size"]))
    root = Path(dcfg.get("root", "."))
    train_df = load_split(Path(dcfg["splits_csv"]), "train", root)
    val_df = load_split(Path(dcfg["splits_csv"]), "val", root)
    if dcfg.get("simulate_defect_prevalence"):
        frac = float(dcfg["simulate_defect_prevalence"])
        train_df = simulate_prevalence(train_df, frac, cfg["seed"])
        val_df = simulate_prevalence(val_df, frac, cfg["seed"])

    cache = bool(dcfg.get("cache_images", True))
    train_ds = DefectDataset(train_df, TrainTransform(pp, cfg.get("augmentation")), cache=cache)
    val_ds = DefectDataset(val_df, EvalTransform(pp), cache=cache)
    counts = train_ds.class_counts()
    log.info("Train class counts %s | val class counts %s", dict(zip(CLASS_NAMES, counts.tolist())),
             dict(zip(CLASS_NAMES, val_ds.class_counts().tolist())))

    strategy = cfg["imbalance"]["strategy"]
    class_weights = None
    sampler = None
    if strategy == "weighted_loss":
        class_weights = inverse_frequency_weights(counts).to(device)
        log.info("Class weights (inverse frequency): %s", class_weights.tolist())
    elif strategy == "weighted_sampler":
        sampler = balanced_sampler(train_ds.labels, cfg["seed"])
    elif strategy != "none":
        raise ValueError(f"unknown imbalance strategy {strategy}")

    nw = int(tcfg.get("num_workers", 0))
    loader_kw = {"num_workers": nw, "persistent_workers": nw > 0, "worker_init_fn": _worker_init}
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=sampler is None, sampler=sampler,
                              drop_last=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"] * 2, shuffle=False, **loader_kw)

    mcfg = cfg["model"]
    model = build_model(mcfg["name"], num_classes=len(CLASS_NAMES), pretrained=mcfg.get("pretrained", True),
                        weights_path=mcfg.get("weights_path"), drop_rate=float(mcfg.get("drop_rate", 0.2))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(tcfg.get("label_smoothing", 0.0)))

    epochs, freeze_epochs = int(tcfg["epochs"]), int(tcfg.get("freeze_epochs", 0))
    steps_per_epoch = len(train_loader)

    def make_optimizer(full: bool):
        if full:
            groups = param_groups(model, tcfg["lr"], tcfg["backbone_lr_mult"], tcfg["weight_decay"])
            n_steps = (epochs - freeze_epochs) * steps_per_epoch
        else:
            groups = [{"params": model.get_classifier().parameters(), "lr": tcfg["lr"],
                       "weight_decay": tcfg["weight_decay"]}]
            n_steps = max(freeze_epochs, 1) * steps_per_epoch
        opt = torch.optim.AdamW(groups)
        warm = int(tcfg.get("warmup_epochs", 0)) * steps_per_epoch if full else 0
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(warm, n_steps))
        return opt, sched

    set_backbone_trainable(model, freeze_epochs == 0)
    optimizer, scheduler = make_optimizer(full=freeze_epochs == 0)

    history = []
    best_loss, best_epoch, bad_epochs = float("inf"), -1, 0
    patience = int(tcfg.get("early_stopping_patience", 1_000))
    ckpt_path = run_dir / "best.pt"
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            log.info("Unfreezing backbone for full fine-tuning")
            set_backbone_trainable(model, True)
            optimizer, scheduler = make_optimizer(full=True)
        model.train()
        t0, run_loss, n_seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
        y_val, p_val, val_loss = predict(model, val_loader, device)
        m = compute_metrics(y_val, p_val, 0.5)
        row = {"epoch": epoch, "train_loss": run_loss / n_seen, "val_loss": val_loss, "val_f1": m["f1"],
               "val_recall": m["recall"], "val_precision": m["precision"], "val_roc_auc": m["roc_auc"],
               "val_pr_auc": m["pr_auc"], "lr_head": optimizer.param_groups[-1]["lr"],
               "epoch_sec": time.time() - t0}
        history.append(row)
        log.info("epoch %02d | train_loss %.4f | val_loss %.4f | val F1 %.4f P %.4f R %.4f AUC %.4f | %.0fs",
                 epoch, row["train_loss"], val_loss, m["f1"], m["precision"], m["recall"], m["roc_auc"] or 0,
                 row["epoch_sec"])
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch}, ckpt_path)
        else:
            bad_epochs += 1
            if epoch > freeze_epochs and bad_epochs >= patience:
                log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break
    train_minutes = (time.time() - t_start) / 60
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Reload best weights, tune the decision threshold on validation, and finalise the checkpoint.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    y_val, p_val, val_loss = predict(model, val_loader, device)
    threshold, policy = select_threshold(y_val, p_val, cfg["imbalance"].get("target_recall"))
    val_metrics = compute_metrics(y_val, p_val, threshold)
    log.info("Best epoch %d | val_loss %.4f | threshold %.4f (%s) | val %s", best_epoch, val_loss, threshold,
             policy, json.dumps({k: val_metrics[k] for k in ("precision", "recall", "f1", "roc_auc")}))
    ckpt.update({
        "class_names": list(CLASS_NAMES),
        "image_size": pp.image_size,
        "mean": list(pp.mean),
        "std": list(pp.std),
        "threshold": threshold,
        "threshold_policy": policy,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "train_class_counts": counts.tolist(),
        "train_minutes": train_minutes,
    })
    torch.save(ckpt, ckpt_path)
    (run_dir / "val_metrics.json").write_text(json.dumps(
        {"threshold": threshold, "policy": policy, "best_epoch": best_epoch, "metrics": val_metrics,
         "train_minutes": train_minutes}, indent=2))
    plot_history(pd.DataFrame(history), run_dir / "training_curves.png", best_epoch)
    log.info("Saved %s (%.1f min)", ckpt_path, train_minutes)
    logging.getLogger().removeHandler(file_handler)
    return ckpt_path


def plot_history(h: pd.DataFrame, out: Path, best_epoch: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(h.epoch, h.train_loss, label="train (weighted, augmented)")
    axes[0].plot(h.epoch, h.val_loss, label="val")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(h.epoch, h.val_f1, label="val F1 (defective, t=0.5)")
    axes[1].plot(h.epoch, h.val_roc_auc, label="val ROC-AUC")
    axes[1].set_title("Validation metrics")
    for ax in axes:
        ax.axvline(best_epoch, ls="--", c="gray", lw=1)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:  # e.g. train.epochs=3  imbalance.strategy=none
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(val)
    return cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
