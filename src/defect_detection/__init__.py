"""Visual defect detection for manufactured parts (normal vs defective)."""

__version__ = "1.0.0"

CLASS_NAMES: tuple[str, ...] = ("normal", "defective")
"""Index 0 = normal, index 1 = defective. The *positive* class for metrics is `defective`."""
