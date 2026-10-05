Train/val defect prevalence sub-sampled to 10%; evaluated on the full test split.

| strategy | threshold | recall | precision | F1 | specificity | FN | FP | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| none | 0.5 | 0.9746 | 1.0000 | 0.9871 | 1.0000 | 3 | 0 | 1.0000 |
| none | 0.246 (tuned) | 0.9915 | 1.0000 | 0.9957 | 1.0000 | 1 | 0 | 1.0000 |
| weighted_loss | 0.5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 1.0000 |
| weighted_loss | 0.540 (tuned) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 1.0000 |
| weighted_sampler | 0.5 | 0.9915 | 1.0000 | 0.9957 | 1.0000 | 1 | 0 | 1.0000 |
| weighted_sampler | 0.491 (tuned) | 0.9915 | 1.0000 | 0.9957 | 1.0000 | 1 | 0 | 1.0000 |
