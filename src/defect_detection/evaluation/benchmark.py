"""Latency / throughput / size benchmark: PyTorch eager vs ONNX Runtime FP32 vs ONNX Runtime INT8.

Also reports test-set F1 / recall / precision of every ONNX variant at the deployed threshold, so the
accuracy cost of each speed-up is visible in one table.

Usage::

    python -m defect_detection.evaluation.benchmark \
        --model efficientnet_b0=models/efficientnet_b0:runs/efficientnet_b0/best.pt \
        --model mobilenetv3_large_100=models/mobilenetv3_large_100:runs/mobilenetv3_large_100/best.pt \
        --out reports/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from defect_detection.data.dataset import load_split
from defect_detection.evaluation.metrics import compute_metrics
from defect_detection.inference import DefectClassifier

log = logging.getLogger("benchmark")


def timeit(fn, n_warmup: int, n_iter: int) -> dict:
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2),
            "mean_ms": round(float(ts.mean()), 2)}


def bench_model(name: str, model_dir: Path, ckpt: Path | None, splits_csv: Path, threads: int,
                n_iter: int) -> list[dict]:
    rows = []
    test = load_split(splits_csv, "test")
    imgs = [Image.open(p).convert("L") for p in test.abs_path]
    y = test.label.to_numpy()
    raw = Image.open(test.abs_path.iloc[0])

    variants = [("onnx-fp32", "model.onnx"), ("onnx-int8", "model.int8.onnx")]
    for vname, fname in variants:
        if not (model_dir / fname).is_file():
            continue
        clf = DefectClassifier(model_dir, onnx_filename=fname, intra_op_threads=threads)
        x1 = clf.preprocess([raw])
        x8 = np.repeat(x1, 8, axis=0)
        pre = timeit(lambda c=clf: c.preprocess([raw]), 3, n_iter)
        b1 = timeit(lambda c=clf, x=x1: c.predict_proba(x), 5, n_iter)
        b8 = timeit(lambda c=clf, x=x8: c.predict_proba(x), 3, max(n_iter // 4, 5))
        p = np.concatenate([clf.predict_proba(clf.preprocess(imgs[i: i + 32]))[:, 1]
                            for i in range(0, len(imgs), 32)])
        m = compute_metrics(y, p, clf.threshold)
        rows.append({
            "model": name, "runtime": vname, "size_mb": round((model_dir / fname).stat().st_size / 1e6, 2),
            "preprocess_ms_p50": pre["p50_ms"], "batch1_p50_ms": b1["p50_ms"], "batch1_p95_ms": b1["p95_ms"],
            "batch8_per_image_ms": round(b8["p50_ms"] / 8, 2),
            "throughput_img_s_batch8": round(8000 / b8["p50_ms"], 1),
            "test_f1": round(m["f1"], 4), "test_recall": round(m["recall"], 4),
            "test_precision": round(m["precision"], 4),
            "test_fn": m["confusion_matrix"]["fn"], "test_fp": m["confusion_matrix"]["fp"],
        })
        log.info("%s", rows[-1])

    if ckpt is not None and ckpt.is_file():
        import torch

        from defect_detection.export.onnx_export import load_checkpoint_model

        torch.set_num_threads(threads)
        model, _ = load_checkpoint_model(ckpt)
        x1 = torch.from_numpy(clf.preprocess([raw]))
        x8 = x1.repeat(8, 1, 1, 1)
        with torch.inference_mode():
            b1 = timeit(lambda: model(x1), 5, n_iter)
            b8 = timeit(lambda: model(x8), 3, max(n_iter // 4, 5))
        size = sum(p.numel() * p.element_size() for p in model.state_dict().values()) / 1e6
        rows.insert(0, {"model": name, "runtime": "pytorch-fp32 (eager)", "size_mb": round(size, 2),
                        "preprocess_ms_p50": pre["p50_ms"], "batch1_p50_ms": b1["p50_ms"],
                        "batch1_p95_ms": b1["p95_ms"], "batch8_per_image_ms": round(b8["p50_ms"] / 8, 2),
                        "throughput_img_s_batch8": round(8000 / b8["p50_ms"], 1)})
        log.info("%s", rows[0])
    return rows


def to_markdown(rows: list[dict]) -> str:
    hdr = ["model", "runtime", "size_mb", "preprocess_ms_p50", "batch1_p50_ms", "batch1_p95_ms", "batch8_per_image_ms",
           "throughput_img_s_batch8", "test_f1", "test_recall", "test_precision", "test_fn", "test_fp"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "–")) for h in hdr) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=MODEL_DIR[:CHECKPOINT]")
    ap.add_argument("--splits-csv", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("reports/benchmark.json"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rows = []
    for spec in args.model:
        name, rest = spec.split("=", 1)
        mdir, _, ck = rest.partition(":")
        rows += bench_model(name, Path(mdir), Path(ck) if ck else None, args.splits_csv, args.threads, args.n_iter)
    env = {"cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(), "threads": args.threads,
           "python": platform.python_version()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"environment": env, "results": rows}, indent=2))
    args.out.with_suffix(".md").write_text(f"Environment: {env}\n\n" + to_markdown(rows) + "\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
