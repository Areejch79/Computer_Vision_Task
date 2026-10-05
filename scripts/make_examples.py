"""Generate ``examples/``: sample images from the *test* split + the API's responses for them.

The requests go through the real FastAPI app (in-process TestClient) with the exported model, so the
JSON shown in the README is exactly what a client receives.

    python scripts/make_examples.py --model-dir models/efficientnet_b0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from defect_detection.api.main import create_app
from defect_detection.api.settings import Settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("models/efficientnet_b0"))
    ap.add_argument("--predictions-csv", type=Path, default=Path("reports/efficientnet_b0/predictions.csv"))
    ap.add_argument("--out", type=Path, default=Path("examples"))
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions_csv)
    picks = pd.concat([
        preds[preds.outcome == "TN"].sort_values("p_defective").head(2),
        preds[preds.outcome == "TP"].sort_values("p_defective", ascending=False).head(2),
        preds[preds.outcome == "TP"].sort_values("p_defective").head(1),  # least confident hit
        preds[preds.outcome == "FN"].head(2),
        preds[preds.outcome == "FP"].head(2),
    ]).drop_duplicates("path")
    img_dir = args.out / "images"
    shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True)

    app = create_app(Settings(model_dir=args.model_dir, log_json=False))
    rows = []
    with TestClient(app) as client:
        for _, r in picks.iterrows():
            src = Path(r.path)
            dst = img_dir / f"{r.class_name}__{src.name}"
            shutil.copy(src, dst)
            with open(dst, "rb") as f:
                resp = client.post("/predict", files={"file": (dst.name, f, "image/jpeg")},
                                   headers={"x-request-id": f"example-{len(rows):02d}"})
            body = resp.json()
            body.pop("inference_ms", None)  # machine dependent; keep the committed file stable
            rows.append({"image": str(dst.relative_to(args.out)), "ground_truth": r.class_name,
                         "outcome": r.outcome, "response": body})
    (args.out / "predictions.json").write_text(json.dumps(rows, indent=2))
    lines = ["# Example predictions", "",
             "Images are from the held-out **test split**. Responses were produced by `POST /predict` of the "
             "API with the shipped model (`scripts/make_examples.py`).", "",
             "| image | ground truth | predicted | confidence | P(defective) | requires_review | outcome |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        b = r["response"]
        lines.append(f"| ![]({r['image']}) `{Path(r['image']).name}` | {r['ground_truth']} "
                     f"| **{b['predicted_class']}** | {b['confidence']:.4f} | {b['defect_probability']:.4f} "
                     f"| {b['requires_review']} | {r['outcome']} |")
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
