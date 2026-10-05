"""Dataset analysis + leakage-safe train/val/test split.

Steps
-----
1. Index every image under ``<raw_dir>/<class_folder>/`` and map folder names to labels.
2. Validate files (decodable, size, mode) and compute per-image statistics used for EDA,
   in particular *border brightness*: a proxy for background/lighting that turned out to differ
   between classes in this dataset (a potential shortcut the model could learn).
3. Find exact duplicates (MD5) and near-duplicates (16x16 difference hash, Hamming distance)
   and merge them into groups with union-find. The split is done at *group* level so that
   near-identical shots of the same part can never end up in both train and test.
4. Stratified group split -> ``data/splits.csv`` (committed, so every run uses the same split).
5. Write ``reports/data_analysis.json`` and EDA figures.

Usage::

    python -m defect_detection.data.prepare --raw-dir data/raw --out data/splits.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from defect_detection import CLASS_NAMES

log = logging.getLogger("prepare")

# Folder name in the raw dataset -> canonical class name.
FOLDER_TO_CLASS = {"ok_front": "normal", "def_front": "defective", "normal": "normal", "defective": "defective"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BORDER_PX = 20


def dhash(gray: Image.Image, n: int = 16) -> np.ndarray:
    small = np.asarray(gray.resize((n + 1, n), Image.Resampling.BILINEAR), dtype=np.float32)
    return (small[:, 1:] > small[:, :-1]).ravel()


def image_stats(path: Path) -> dict:
    raw = path.read_bytes()
    with Image.open(path) as img:
        width, height, mode = img.width, img.height, img.mode
        gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    b = BORDER_PX
    border = np.concatenate([arr[:b].ravel(), arr[-b:].ravel(), arr[:, :b].ravel(), arr[:, -b:].ravel()])
    if mode == "RGB":
        with Image.open(path) as img:
            rgb = np.asarray(img, dtype=np.int16)
        is_gray = bool(np.abs(rgb[..., 0] - rgb[..., 1]).max() <= 2 and np.abs(rgb[..., 1] - rgb[..., 2]).max() <= 2)
    else:
        is_gray = True
    # Variance of the Laplacian = cheap sharpness / blur indicator.
    lap = (
        -4 * arr[1:-1, 1:-1] + arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:]
    )
    return {
        "width": width,
        "height": height,
        "mode": mode,
        "is_grayscale_content": is_gray,
        "file_kb": round(len(raw) / 1024, 1),
        "md5": hashlib.md5(raw).hexdigest(),
        "mean_brightness": float(arr.mean()),
        "std_brightness": float(arr.std()),
        "border_brightness": float(border.mean()),
        "sharpness": float(lap.var()),
        "_dhash": dhash(gray),
    }


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def assign_groups(df: pd.DataFrame, hashes: np.ndarray, max_hamming: int) -> pd.Series:
    uf = UnionFind(len(df))
    for _, idx in df.groupby("md5").indices.items():
        for k in idx[1:]:
            uf.union(idx[0], k)
    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    ii, jj = np.where(np.triu(dist <= max_hamming, k=1))
    for i, j in zip(ii, jj):
        uf.union(int(i), int(j))
    return pd.Series([uf.find(i) for i in range(len(df))], index=df.index)


def stratified_group_split(
    df: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> pd.Series:
    """Assign whole groups to splits, per class, so class ratios are preserved in every split.

    A group whose members carry *different* labels (possible only for near-duplicates across
    classes) is assigned by its majority label and logged; such groups are worth a manual look.
    """
    rng = np.random.default_rng(seed)
    group_label = df.groupby("group")["label"].agg(lambda s: int(s.mode().iloc[0]))
    mixed = df.groupby("group")["label"].nunique()
    for g in mixed[mixed > 1].index:
        log.warning("Group %s contains near-duplicates with different labels: %s", g,
                    df.loc[df.group == g, "path"].tolist())
    split_of_group: dict[int, str] = {}
    for label in sorted(group_label.unique()):
        groups = group_label[group_label == label].index.to_numpy().copy()
        rng.shuffle(groups)
        sizes = df[df.group.isin(groups)].groupby("group").size().loc[groups].to_numpy()
        total = sizes.sum()
        cum = np.cumsum(sizes)
        n_test_end = np.searchsorted(cum, test_frac * total) + 1
        n_val_end = np.searchsorted(cum, (test_frac + val_frac) * total) + 1
        for k, g in enumerate(groups):
            split_of_group[g] = "test" if k < n_test_end else "val" if k < n_val_end else "train"
    return df["group"].map(split_of_group)


def save_figures(df: pd.DataFrame, raw_dir: Path, fig_dir: Path, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"normal": "#2a78b5", "defective": "#d1495b"}

    # 1) Class distribution per split
    fig, ax = plt.subplots(figsize=(6, 3.5))
    tab = df.groupby(["split", "class_name"]).size().unstack(fill_value=0).loc[["train", "val", "test"]]
    tab = tab[list(CLASS_NAMES)]
    tab.plot(kind="bar", ax=ax, color=[colors[c] for c in tab.columns], rot=0)
    for cont in ax.containers:
        ax.bar_label(cont, fontsize=8)
    ax.set_title("Images per split and class")
    ax.set_ylabel("images")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=120)
    plt.close(fig)

    # 2) Brightness distributions (shortcut check)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, col, title in [
        (axes[0], "border_brightness", "Background (border) brightness"),
        (axes[1], "mean_brightness", "Mean image brightness"),
    ]:
        for c in CLASS_NAMES:
            ax.hist(df.loc[df.class_name == c, col], bins=40, alpha=0.6, label=c, color=colors[c])
        ax.set_title(title)
        ax.set_xlabel("gray level (0-255)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "brightness_by_class.png", dpi=120)
    plt.close(fig)

    # 3) Sample grid
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for r, c in enumerate(CLASS_NAMES):
        paths = df.loc[df.class_name == c, "path"].to_numpy()
        for k, p in enumerate(rng.choice(paths, 6, replace=False)):
            axes[r, k].imshow(Image.open(raw_dir.parent.parent / p).convert("L"), cmap="gray")
            axes[r, k].set_title(f"{c}\n{Path(p).name}", fontsize=7)
            axes[r, k].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "samples.png", dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--near-dup-hamming", type=int, default=10,
                    help="max Hamming distance (of 256 bits) between dHashes to treat images as near-duplicates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = args.raw_dir.resolve().parent.parent  # paths in the CSV are relative to the repo root
    rows, hashes, bad = [], [], []
    for folder in sorted(p for p in args.raw_dir.iterdir() if p.is_dir()):
        cls = FOLDER_TO_CLASS.get(folder.name)
        if cls is None:
            log.warning("Skipping unknown folder %s", folder)
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                st = image_stats(path)
            except Exception as exc:  # corrupt file -> exclude, but report it
                bad.append({"path": str(path), "error": repr(exc)})
                continue
            hashes.append(st.pop("_dhash"))
            rows.append({"path": str(path.resolve().relative_to(root)), "class_name": cls,
                         "label": CLASS_NAMES.index(cls), **st})
    if not rows:
        raise SystemExit(f"No images found under {args.raw_dir}. Run scripts/download_data.sh first.")
    df = pd.DataFrame(rows)
    H = np.stack(hashes)
    log.info("Indexed %d images (%d unreadable)", len(df), len(bad))

    df["group"] = assign_groups(df, H, args.near_dup_hamming)
    df["split"] = stratified_group_split(df, args.val_frac, args.test_frac, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["path", "class_name", "label", "group", "split"]].to_csv(args.out, index=False)
    log.info("Wrote %s", args.out)

    group_sizes = df.groupby("group").size()
    by_class = df.groupby("class_name")
    analysis = {
        "n_images": int(len(df)),
        "unreadable_files": bad,
        "class_counts": df.class_name.value_counts().to_dict(),
        "class_ratio_defective": round(float((df.label == 1).mean()), 4),
        "image_sizes": (df.width.astype(str) + "x" + df.height.astype(str)).value_counts().to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "all_rgb_files_have_identical_channels": bool(df.is_grayscale_content.all()),
        "exact_duplicates": int(len(df) - df.md5.nunique()),
        "near_duplicate_groups": int((group_sizes > 1).sum()),
        "images_in_near_duplicate_groups": int(group_sizes[group_sizes > 1].sum()),
        "near_dup_hamming_threshold": args.near_dup_hamming,
        "split_counts": {
            s: df[df.split == s].class_name.value_counts().to_dict() for s in ["train", "val", "test"]
        },
        "per_class_stats": {
            c: {
                k: {"mean": round(float(g[k].mean()), 2), "std": round(float(g[k].std()), 2)}
                for k in ["mean_brightness", "border_brightness", "std_brightness", "sharpness", "file_kb"]
            }
            for c, g in by_class
        },
    }
    # How separable are the classes from *background brightness alone*? (shortcut indicator)
    from sklearn.metrics import roc_auc_score

    analysis["shortcut_auc_border_brightness"] = round(
        float(1 - roc_auc_score(df.label, df.border_brightness)), 4
    )  # darker background => defective, so invert
    analysis["shortcut_auc_mean_brightness"] = round(float(1 - roc_auc_score(df.label, df.mean_brightness)), 4)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "data_analysis.json").write_text(json.dumps(analysis, indent=2))
    save_figures(df, args.raw_dir, args.report_dir / "figures" / "data", args.seed)
    log.info("Data analysis:\n%s", json.dumps({k: v for k, v in analysis.items() if k != "per_class_stats"}, indent=2))


if __name__ == "__main__":
    main()
