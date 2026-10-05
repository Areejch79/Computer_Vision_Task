"""Class-imbalance ablation.

The provided data is 60 % defective, the opposite of a real line where defects are rare. To study
imbalance handling under realistic conditions we sub-sample defects in *train and val* to 10 %
prevalence and compare three strategies (MobileNetV3 for speed):

* ``none``              – plain cross-entropy
* ``weighted_loss``     – inverse-frequency class weights in the loss
* ``weighted_sampler``  – class-balanced mini-batches (oversampling the minority class)

Each model is evaluated on the untouched test split at threshold 0.5 *and* at the threshold tuned on
(imbalanced) validation, because threshold moving is itself an imbalance remedy.

    python scripts/imbalance_ablation.py --epochs 12
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import DefectDataset, load_split
from defect_detection.data.transforms import EvalTransform
from defect_detection.evaluation.metrics import compute_metrics
from defect_detection.export.onnx_export import load_checkpoint_model
from defect_detection.preprocessing import PreprocessConfig
from defect_detection.training.train import load_config, predict, train

STRATEGIES = ["none", "weighted_loss", "weighted_sampler"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/mobilenetv3_large.yaml"))
    ap.add_argument("--prevalence", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("reports/imbalance_ablation.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    results = []
    for strat in STRATEGIES:
        cfg = load_config(args.config, [
            f"run_name=ablation_{strat}", f"imbalance.strategy={strat}",
            f"data.simulate_defect_prevalence={args.prevalence}", f"train.epochs={args.epochs}",
            "train.early_stopping_patience=5",
        ])
        ckpt_path = train(cfg)
        model, ckpt = load_checkpoint_model(ckpt_path)
        pp = PreprocessConfig(image_size=ckpt["image_size"])
        test = DefectDataset(load_split(Path(cfg["data"]["splits_csv"]), "test"), EvalTransform(pp))
        y, p, _ = predict(model, DataLoader(test, batch_size=64), torch.device("cpu"))
        row = {"strategy": strat, "train_counts": ckpt["train_class_counts"], "tuned_threshold": ckpt["threshold"]}
        for name, t in [("t=0.5", 0.5), ("tuned", ckpt["threshold"])]:
            m = compute_metrics(y, p, t)
            row[name] = {k: round(m[k], 4) for k in ("precision", "recall", "f1", "specificity", "balanced_accuracy")}
            row[name].update(m["confusion_matrix"])
        row["roc_auc"] = round(compute_metrics(y, p)["roc_auc"], 4)
        row["mean_p_defective_on_true_defects"] = round(float(np.mean(p[y == 1])), 4)
        results.append(row)
        logging.info("ablation %s -> %s", strat, row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"prevalence_in_train_val": args.prevalence, "results": results}, indent=2))
    lines = [f"Train/val defect prevalence sub-sampled to {args.prevalence:.0%}; evaluated on the full test split.", "",
             "| strategy | threshold | recall | precision | F1 | specificity | FN | FP | ROC-AUC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for name in ("t=0.5", "tuned"):
            m = r[name]
            t = "0.5" if name == "t=0.5" else f"{r['tuned_threshold']:.3f} (tuned)"
            lines.append(f"| {r['strategy']} | {t} | {m['recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | "
                         f"{m['specificity']:.4f} | {m['fn']} | {m['fp']} | {r['roc_auc']:.4f} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
