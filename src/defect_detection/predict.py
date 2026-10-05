"""Command-line inference (no server needed)::

    python -m defect_detection.predict --model-dir models/efficientnet_b0 path/to/img1.jpg path/to/dir/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from defect_detection.inference import DefectClassifier

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="image files or directories")
    ap.add_argument("--model-dir", type=Path, default=Path("models/efficientnet_b0"))
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args(argv)
    clf = DefectClassifier(args.model_dir, threshold=args.threshold)
    paths = [q for p in args.inputs for q in (sorted(p.rglob("*")) if p.is_dir() else [p]) if q.suffix.lower() in EXTS]
    exit_code = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                (pred,), ms = clf.predict([im])
            print(json.dumps({"file": str(p), **pred.to_dict(), "inference_ms": round(ms, 2)}))
        except Exception as exc:  # keep going on bad files, report them on stderr
            print(json.dumps({"file": str(p), "error": repr(exc)}), file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
