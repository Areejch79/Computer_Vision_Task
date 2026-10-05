"""Evaluate a *deployed* (ONNX) model on the held-out test split + error analysis.

We evaluate the exact artefact the API serves (ONNX + metadata threshold), not the training
checkpoint, so reported numbers are the numbers production gets.

Outputs (``--out-dir``):

* ``metrics.json``            – metrics at the tuned threshold and at 0.5, bootstrap 95 % CIs,
                                threshold sweep, robustness + shortcut analysis
* ``predictions.csv``         – per-image probabilities, predictions, image statistics
* ``error_analysis.md``       – human-readable summary of FPs / FNs and model weaknesses
* ``figures/*.png``           – confusion matrices, ROC / PR curves, score histogram,
                                reliability diagram, FP/FN galleries with Grad-CAM

Usage::

    python -m defect_detection.evaluation.evaluate --model-dir models/efficientnet_b0 \
        --checkpoint runs/efficientnet_b0/best.pt --out-dir reports/efficientnet_b0
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from defect_detection.data.dataset import load_split
from defect_detection.evaluation.metrics import compute_metrics
from defect_detection.inference import DefectClassifier
from defect_detection.preprocessing import resize, to_model_mode

log = logging.getLogger("evaluate")
BORDER = 20


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def load_gray(path: str) -> Image.Image:
    with Image.open(path) as im:
        g = to_model_mode(im)
        g.load()
    return g


def image_features(img: Image.Image) -> dict:
    a = np.asarray(img, dtype=np.float32)
    border = np.concatenate([a[:BORDER].ravel(), a[-BORDER:].ravel(), a[:, :BORDER].ravel(), a[:, -BORDER:].ravel()])
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return {"border_brightness": float(border.mean()), "mean_brightness": float(a.mean()),
            "contrast": float(a.std()), "sharpness": float(lap.var())}


def batched_proba(clf: DefectClassifier, images: list[Image.Image], bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(images), bs):
        out.append(clf.predict_proba(clf.preprocess(images[i: i + bs]))[:, clf.positive_index])
    return np.concatenate(out)


def bootstrap_ci(y: np.ndarray, p: np.ndarray, t: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile bootstrap 95 % CI – the test set is small, so point estimates alone mislead."""
    rng = np.random.default_rng(seed)
    keys = ["precision", "recall", "f1", "specificity", "accuracy"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = compute_metrics(y[idx], p[idx], t)
        for k in keys:
            samples[k].append(m[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in samples.items()}


def prevalence_adjusted_precision(m: dict, prevalences=(0.01, 0.05, 0.10)) -> dict:
    """Precision depends on the defect rate; recall and FPR do not.

    The dataset is 60 % defective but a production line typically has a few % defects, so we
    re-weight the measured recall/FPR to show what precision (share of rejected parts that are
    really defective) the line would see:  P = R*pi / (R*pi + FPR*(1-pi)).
    """
    r, fpr = m["recall"], m["false_positive_rate"]
    out = {}
    for pi in prevalences:
        denom = r * pi + fpr * (1 - pi)
        out[f"{pi:.0%}"] = round(r * pi / denom, 4) if denom > 0 else None
    return out


# --------------------------------------------------------------------------------------------
# perturbations for robustness testing
# --------------------------------------------------------------------------------------------
def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def _noise(img: Image.Image, std: float, seed: int) -> Image.Image:
    a = np.asarray(img, dtype=np.float32)
    a = a + np.random.default_rng(seed).normal(0, std, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L")


def _rotate(img: Image.Image, deg: float) -> Image.Image:
    # fill with the image's own background level so we do not add an artificial cue
    a = np.asarray(img)
    fill = int(np.median(np.concatenate([a[:BORDER].ravel(), a[-BORDER:].ravel()])))
    return img.rotate(deg, resample=Image.Resampling.BILINEAR, fillcolor=fill)


def _center_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


PERTURBATIONS = {
    "brightness x0.7": lambda im, s: ImageEnhance.Brightness(im).enhance(0.7),
    "brightness x1.3": lambda im, s: ImageEnhance.Brightness(im).enhance(1.3),
    "contrast x0.7": lambda im, s: ImageEnhance.Contrast(im).enhance(0.7),
    "gaussian blur r=2": lambda im, s: im.filter(ImageFilter.GaussianBlur(2)),
    "gaussian noise std=10": lambda im, s: _noise(im, 10, s),
    "jpeg quality 30": lambda im, s: _jpeg(im, 30),
    "rotation 45 deg": lambda im, s: _rotate(im, 45),
    "zoom (center crop 90%)": lambda im, s: _center_crop(im, 0.9),
}


def _downscale(img: Image.Image, px: int) -> Image.Image:
    """Simulate a lower-resolution camera: downsample to ``px`` and back to the original size."""
    w, h = img.size
    return img.resize((px, px), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)


# Severity sweeps used to find *where* the model breaks (the test set itself is too easy to show it).
STRESS = {
    "gaussian_blur_radius": ([1, 2, 3, 4, 6], lambda im, v, s: im.filter(ImageFilter.GaussianBlur(v))),
    "brightness_factor": ([0.4, 0.6, 0.8, 1.2, 1.4, 1.6], lambda im, v, s: ImageEnhance.Brightness(im).enhance(v)),
    "gaussian_noise_std": ([5, 10, 20, 30, 50], lambda im, v, s: _noise(im, v, s)),
    "jpeg_quality": ([50, 20, 10, 5], lambda im, v, s: _jpeg(im, v)),
    "camera_resolution_px": ([256, 192, 128, 96, 64], lambda im, v, s: _downscale(im, v)),
}


def shift_background(img: Image.Image, target_border: float) -> Image.Image:
    """Counterfactual: additive intensity shift so that the background level equals ``target_border``."""
    a = np.asarray(img, dtype=np.float32)
    cur = image_features(img)["border_brightness"]
    return Image.fromarray(np.clip(a + (target_border - cur), 0, 255).astype(np.uint8), mode="L")


def run_stress(clf: DefectClassifier, images: list[Image.Image], y: np.ndarray, t: float):
    out, first_failures = {}, None
    for name, (levels, fn) in STRESS.items():
        out[name] = []
        for v in levels:
            perturbed = [fn(im, v, i) for i, im in enumerate(images)]
            pv = batched_proba(clf, perturbed)
            m = compute_metrics(y, pv, t)
            flagged = np.array([bool(clf.check_quality(resize(im, clf.pp.image_size))) for im in perturbed])
            fn_mask = (y == 1) & (pv < t)
            out[name].append({"level": v, **{k: round(m[k], 4) for k in ("recall", "specificity", "f1")},
                              "fn": m["confusion_matrix"]["fn"], "fp": m["confusion_matrix"]["fp"],
                              "quality_flagged_rate": round(float(flagged.mean()), 4),
                              "fn_not_flagged": int((fn_mask & ~flagged).sum())})
            if name == "gaussian_blur_radius" and first_failures is None and m["confusion_matrix"]["fn"] > 0:
                first_failures = (name, v, np.where((y == 1) & (pv < t))[0][:6])
        log.info("stress %-22s %s", name, [(r["level"], r["f1"]) for r in out[name]])
    return out, first_failures


def plot_stress(stress: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(stress), figsize=(3.4 * len(stress), 3.2))
    for ax, (name, rows) in zip(axes, stress.items()):
        lv = [str(r["level"]) for r in rows]
        ax.plot(lv, [r["recall"] for r in rows], "o-", label="recall (defects caught)", c="#d1495b")
        ax.plot(lv, [r["specificity"] for r in rows], "s-", label="specificity (normals passed)", c="#2a78b5")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Stress test: degradation vs severity (tuned threshold)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------
def make_figures(df: pd.DataFrame, m_tuned: dict, m_05: dict, threshold: float, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig_dir.mkdir(parents=True, exist_ok=True)
    y, p = df.label.to_numpy(), df.p_defective.to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, m, title in [(axes[0], m_tuned, f"tuned threshold = {threshold:.3f}"), (axes[1], m_05, "threshold = 0.5")]:
        cm = m["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                    color="white" if v > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["normal", "defective"])
        ax.set_yticks([0, 1], ["normal", "defective"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(title)
    fig.suptitle("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, lw=2)
    axes[0].plot([0, 1], [0, 1], "--", c="gray")
    axes[0].scatter([m_tuned["false_positive_rate"]], [m_tuned["recall"]], c="red", zorder=3, label="operating point")
    axes[0].set(title=f"ROC (AUC = {m_tuned['roc_auc']:.4f})", xlabel="FPR", ylabel="TPR (recall)")
    axes[0].legend()
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, lw=2)
    axes[1].scatter([m_tuned["recall"]], [m_tuned["precision"]], c="red", zorder=3, label="operating point")
    axes[1].set(title=f"Precision-Recall (AP = {m_tuned['pr_auc']:.4f})", xlabel="recall", ylabel="precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_pr_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, label="actual normal", color="#2a78b5")
    ax.hist(p[y == 1], bins=bins, alpha=0.6, label="actual defective", color="#d1495b")
    ax.axvline(threshold, c="k", ls="--", label=f"threshold {threshold:.3f}")
    ax.set_yscale("log")
    ax.set(xlabel="P(defective)", ylabel="images (log)", title="Score distribution on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "score_histogram.png", dpi=120)
    plt.close(fig)

    # reliability diagram
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, 6)
    xs, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf >= lo) & (conf <= hi)
        if msk.any():
            xs.append(conf[msk].mean())
            accs.append(correct[msk].mean())
            ns.append(int(msk.sum()))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0.5, 1], [0.5, 1], "--", c="gray")
    ax.plot(xs, accs, "o-")
    for x_, a_, n_ in zip(xs, accs, ns):
        ax.annotate(str(n_), (x_, a_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set(xlabel="mean confidence", ylabel="accuracy", title=f"Reliability (ECE = {m_05['ece']:.3f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=120)
    plt.close(fig)


def error_gallery(rows: pd.DataFrame, title: str, out: Path, checkpoint: Path | None, image_size: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = rows.head(12)
    if rows.empty:
        return
    cam_fn = None
    if checkpoint is not None and checkpoint.is_file():
        import torch

        from defect_detection.evaluation.gradcam import GradCAM, overlay
        from defect_detection.export.onnx_export import load_checkpoint_model
        from defect_detection.preprocessing import PreprocessConfig, preprocess_pil

        model, ckpt = load_checkpoint_model(checkpoint)
        pp = PreprocessConfig(image_size=image_size, mean=tuple(ckpt["mean"]), std=tuple(ckpt["std"]))
        cam = GradCAM(model)

        def cam_fn(img):
            x = torch.from_numpy(preprocess_pil(img, pp))[None]
            return overlay(img, cam(x, class_idx=1))

    ncols = 2 if cam_fn else 1
    n = len(rows)
    fig, axes = plt.subplots(n, ncols, figsize=(3.2 * ncols, 3.2 * n), squeeze=False)
    for r, (_, row) in enumerate(rows.iterrows()):
        img = load_gray(row.abs_path)
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(f"{Path(row.path).name}\ntrue={row.class_name} P(def)={row.p_defective:.3f}", fontsize=8)
        if cam_fn:
            axes[r, 1].imshow(cam_fn(img))
            axes[r, 1].set_title("Grad-CAM (defective)", fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------------------------
def evaluate(model_dir: Path, out_dir: Path, splits_csv: Path, checkpoint: Path | None, split: str = "test",
             onnx_filename: str = "model.onnx", skip_robustness: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    clf = DefectClassifier(model_dir, onnx_filename=onnx_filename)
    t = clf.threshold
    df = load_split(splits_csv, split)
    images = [load_gray(p) for p in df.abs_path]
    df["p_defective"] = batched_proba(clf, images)
    df["pred_label"] = (df.p_defective >= t).astype(int)
    df["pred_class"] = np.where(df.pred_label == 1, "defective", "normal")
    df["outcome"] = np.select(
        [(df.label == 1) & (df.pred_label == 1), (df.label == 0) & (df.pred_label == 0),
         (df.label == 0) & (df.pred_label == 1), (df.label == 1) & (df.pred_label == 0)],
        ["TP", "TN", "FP", "FN"], default="?")
    feats = pd.DataFrame([image_features(im) for im in images])
    df = pd.concat([df, feats], axis=1)
    y, p = df.label.to_numpy(), df.p_defective.to_numpy()

    m_tuned = compute_metrics(y, p, t)
    m_05 = compute_metrics(y, p, 0.5)
    sweep = []
    for th in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        mm = compute_metrics(y, p, th)
        sweep.append({"threshold": th, **{k: round(mm[k], 4) for k in ("precision", "recall", "f1", "specificity")},
                      **mm["confusion_matrix"]})
    results: dict = {
        "model_dir": str(model_dir),
        "onnx_file": onnx_filename,
        "model_version": clf.version,
        "split": split,
        "n_images": int(len(df)),
        "class_counts": df.class_name.value_counts().to_dict(),
        "threshold": t,
        "metrics_at_tuned_threshold": m_tuned,
        "metrics_at_0.5": m_05,
        "bootstrap_95ci_tuned_threshold": bootstrap_ci(y, p, t),
        "expected_precision_at_line_prevalence": prevalence_adjusted_precision(m_tuned),
        "threshold_sweep": sweep,
    }

    # ---------------- shortcut analysis -----------------
    from sklearn.linear_model import LogisticRegression

    train_df = load_split(splits_csv, "train")
    train_feats = pd.DataFrame([image_features(load_gray(pth)) for pth in train_df.abs_path])
    cols = ["border_brightness", "mean_brightness", "contrast"]
    lr = LogisticRegression(max_iter=1000).fit(train_feats[cols], train_df.label)
    p_lr = lr.predict_proba(df[cols])[:, 1]
    results["shortcut_baseline_brightness_only"] = {
        "features": cols,
        "metrics_at_0.5": {k: round(v, 4) for k, v in compute_metrics(y, p_lr, 0.5).items()
                           if k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        "note": "A logistic regression on global brightness statistics only. If this were close to the CNN, "
                "the CNN could be exploiting lighting instead of defects.",
    }
    bg_normal = float(train_feats.loc[train_df.label.values == 0, "border_brightness"].mean())
    bg_def = float(train_feats.loc[train_df.label.values == 1, "border_brightness"].mean())
    cf_images = [shift_background(im, bg_normal if lab == 1 else bg_def) for im, lab in zip(images, y)]
    p_cf = batched_proba(clf, cf_images)
    flips = (p_cf >= t) != (p >= t)
    results["counterfactual_background_swap"] = {
        "description": "Each test image is shifted so its background brightness matches the *other* class's mean "
                       f"(normal bg {bg_normal:.1f}, defective bg {bg_def:.1f}). Prediction flips indicate reliance "
                       "on the lighting shortcut.",
        "prediction_flip_rate": round(float(flips.mean()), 4),
        "flips_normal_to_defective": int(((p < t) & (p_cf >= t)).sum()),
        "flips_defective_to_normal": int(((p >= t) & (p_cf < t)).sum()),
        "metrics": {k: round(v, 4) for k, v in compute_metrics(y, p_cf, t).items()
                    if k in ("accuracy", "precision", "recall", "f1")},
    }

    # ---------------- robustness -----------------
    if not skip_robustness:
        rob = {"clean": {k: round(m_tuned[k], 4) for k in ("precision", "recall", "f1", "specificity")}}
        for name, fn in PERTURBATIONS.items():
            pp_ = batched_proba(clf, [fn(im, i) for i, im in enumerate(images)])
            mm = compute_metrics(y, pp_, t)
            rob[name] = {k: round(mm[k], 4) for k in ("precision", "recall", "f1", "specificity")}
            rob[name].update({"fn": mm["confusion_matrix"]["fn"], "fp": mm["confusion_matrix"]["fp"]})
            log.info("robustness %-24s F1 %.4f R %.4f P %.4f", name, mm["f1"], mm["recall"], mm["precision"])
        results["robustness_at_tuned_threshold"] = rob
        stress, first_failures = run_stress(clf, images, y, t)
        results["stress_test"] = stress
        plot_stress(stress, fig_dir / "stress_curves.png")
        # Which defects are lost first when the image gets blurrier? (typically the smallest defects)
        if first_failures:
            name, level, idx = first_failures
            rows = df.iloc[idx].copy()
            rows["p_defective"] = df.p_defective.iloc[idx]
            error_gallery(rows, f"First failures under {name}={level} (shown clean, with Grad-CAM)",
                          fig_dir / "first_failures_under_stress.png", checkpoint, clf.pp.image_size)
            results["first_failures_under_stress"] = {"condition": f"{name}={level}",
                                                      "images": df.path.iloc[idx].tolist()}

    # ---------------- error analysis -----------------
    errs = df[df.outcome.isin(["FP", "FN"])].copy()
    errs["margin"] = (errs.p_defective - t).abs()
    errs = errs.sort_values("margin", ascending=False)
    near = df[(df.p_defective - t).abs() < 0.15].sort_values("p_defective")
    group_stats = df.groupby("outcome")[["p_defective", "border_brightness", "mean_brightness", "contrast",
                                         "sharpness"]].mean().round(3)
    results["error_analysis"] = {
        "false_positives": errs[errs.outcome == "FP"][["path", "p_defective", "border_brightness"]].round(4)
        .to_dict("records"),
        "false_negatives": errs[errs.outcome == "FN"][["path", "p_defective", "border_brightness"]].round(4)
        .to_dict("records"),
        "n_near_threshold(|p-t|<0.15)": int(len(near)),
        "mean_stats_by_outcome": group_stats.to_dict("index"),
    }
    df.drop(columns=["abs_path"]).to_csv(out_dir / "predictions.csv", index=False)
    make_figures(df, m_tuned, m_05, t, fig_dir)
    error_gallery(errs[errs.outcome == "FN"], "False negatives (missed defects)", fig_dir / "false_negatives.png",
                  checkpoint, clf.pp.image_size)
    error_gallery(errs[errs.outcome == "FP"], "False positives (false alarms)", fig_dir / "false_positives.png",
                  checkpoint, clf.pp.image_size)
    # Hardest correctly classified parts (smallest margin to the threshold) per class.
    hardest = pd.concat([df[df.label == 1].nsmallest(3, "p_defective"), df[df.label == 0].nlargest(3, "p_defective")])
    error_gallery(hardest, "Hardest test images (closest to the threshold)", fig_dir / "hardest_examples.png",
                  checkpoint, clf.pp.image_size)
    results["hardest_examples"] = hardest[["path", "class_name", "p_defective"]].round(4).to_dict("records")
    # Grad-CAM sanity check on a few confident true positives: does the model look at the part?
    tps = df[df.outcome == "TP"].sort_values("p_defective", ascending=False).iloc[::15].head(4)
    error_gallery(tps, "Sample true positives (sanity check)", fig_dir / "true_positives_gradcam.png", checkpoint,
                  clf.pp.image_size)

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (out_dir / "error_analysis.md").write_text(render_markdown(results, group_stats))
    log.info("Test (t=%.3f): P %.4f R %.4f F1 %.4f | FP %d FN %d | ROC-AUC %.4f", t, m_tuned["precision"],
             m_tuned["recall"], m_tuned["f1"], m_tuned["confusion_matrix"]["fp"], m_tuned["confusion_matrix"]["fn"],
             m_tuned["roc_auc"])
    return results


def render_markdown(r: dict, group_stats: pd.DataFrame) -> str:
    m, ci = r["metrics_at_tuned_threshold"], r["bootstrap_95ci_tuned_threshold"]
    cm = m["confusion_matrix"]
    lines = [
        f"# Error analysis – `{r['model_version']}` on the {r['split']} split ({r['n_images']} images)",
        "",
        f"Decision threshold (tuned on validation): **{r['threshold']:.4f}**",
        "",
        "| metric | value | bootstrap 95% CI |",
        "|---|---|---|",
    ]
    for k in ("precision", "recall", "f1", "specificity", "accuracy"):
        lines.append(f"| {k} | {m[k]:.4f} | {ci[k][0]:.3f} – {ci[k][1]:.3f} |")
    lines += [
        f"| ROC-AUC | {m['roc_auc']:.4f} | |",
        f"| PR-AUC | {m['pr_auc']:.4f} | |",
        "",
        f"Confusion matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}",
        "",
        "Expected precision if the line had a lower defect rate (recall/FPR held fixed): "
        + ", ".join(f"{k}: {v}" for k, v in r["expected_precision_at_line_prevalence"].items()),
        "",
        "## False negatives (missed defects – the costly error)",
        "",
    ]
    fns = r["error_analysis"]["false_negatives"]
    lines += [f"- `{e['path']}` P(def)={e['p_defective']:.3f}, background={e['border_brightness']:.1f}" for e in fns] \
        or ["- none"]
    lines += ["", "## False positives (false alarms)", ""]
    fps = r["error_analysis"]["false_positives"]
    lines += [f"- `{e['path']}` P(def)={e['p_defective']:.3f}, background={e['border_brightness']:.1f}" for e in fps] \
        or ["- none"]
    lines += ["", "## Mean image statistics by outcome", "", group_stats.to_markdown(), ""]
    sb = r["shortcut_baseline_brightness_only"]["metrics_at_0.5"]
    cf = r["counterfactual_background_swap"]
    lines += [
        "## Shortcut analysis",
        "",
        f"- Brightness-only logistic regression: accuracy {sb['accuracy']:.3f}, F1 {sb['f1']:.3f}, "
        f"ROC-AUC {sb['roc_auc']:.3f}.",
        f"- Background-swap counterfactual: {cf['prediction_flip_rate'] * 100:.1f}% of predictions flip "
        f"({cf['flips_normal_to_defective']} normal->defective, {cf['flips_defective_to_normal']} defective->normal); "
        f"F1 under swap {cf['metrics']['f1']:.4f}.",
        "",
    ]
    if "robustness_at_tuned_threshold" in r:
        lines += ["## Robustness (tuned threshold)", "", "| condition | precision | recall | F1 | FP | FN |",
                  "|---|---|---|---|---|---|"]
        for k, v in r["robustness_at_tuned_threshold"].items():
            lines.append(f"| {k} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | "
                         f"{v.get('fp', cm['fp'])} | {v.get('fn', cm['fn'])} |")
    if "stress_test" in r:
        lines += ["", "## Stress test (severity sweeps)", "",
                  "| perturbation | level | recall | specificity | F1 | FN | FP | quality-gate flagged "
                  "| FN not flagged |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, rows in r["stress_test"].items():
            for row in rows:
                lines.append(f"| {name} | {row['level']} | {row['recall']:.4f} | {row['specificity']:.4f} | "
                             f"{row['f1']:.4f} | {row['fn']} | {row['fp']} | "
                             f"{row.get('quality_flagged_rate', 0):.0%} | {row.get('fn_not_flagged', '–')} |")
    if "hardest_examples" in r:
        lines += ["", "## Hardest test images", ""]
        lines += [f"- `{h['path']}` ({h['class_name']}) P(def)={h['p_defective']:.4f}" for h in r["hardest_examples"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--splits-csv", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--checkpoint", type=Path, default=None, help="PyTorch checkpoint for Grad-CAM (optional)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--onnx-file", default="model.onnx")
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluate(args.model_dir, args.out_dir, args.splits_csv, args.checkpoint, args.split, args.onnx_file,
             args.skip_robustness)


if __name__ == "__main__":
    main()
