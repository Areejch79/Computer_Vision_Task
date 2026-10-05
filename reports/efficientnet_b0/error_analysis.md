# Error analysis – `efficientnet_b0-20261005-f93e5afb` on the test split (196 images)

Decision threshold (tuned on validation): **0.4617**

| metric | value | bootstrap 95% CI |
|---|---|---|
| precision | 1.0000 | 1.000 – 1.000 |
| recall | 1.0000 | 1.000 – 1.000 |
| f1 | 1.0000 | 1.000 – 1.000 |
| specificity | 1.0000 | 1.000 – 1.000 |
| accuracy | 1.0000 | 1.000 – 1.000 |
| ROC-AUC | 1.0000 | |
| PR-AUC | 1.0000 | |

Confusion matrix: TN=78 FP=0 FN=0 TP=118

Expected precision if the line had a lower defect rate (recall/FPR held fixed): 1%: 1.0, 5%: 1.0, 10%: 1.0

## False negatives (missed defects – the costly error)

- none

## False positives (false alarms)

- none

## Mean image statistics by outcome

| outcome   |   p_defective |   border_brightness |   mean_brightness |   contrast |   sharpness |
|:----------|--------------:|--------------------:|------------------:|-----------:|------------:|
| TN        |         0.016 |             194.097 |           149.627 |     62.416 |     141.893 |
| TP        |         0.962 |             168.432 |           139.219 |     60.139 |     164.013 |

## Shortcut analysis

- Brightness-only logistic regression: accuracy 0.750, F1 0.797, ROC-AUC 0.791.
- Background-swap counterfactual: 0.0% of predictions flip (0 normal->defective, 0 defective->normal); F1 under swap 1.0000.

## Robustness (tuned threshold)

| condition | precision | recall | F1 | FP | FN |
|---|---|---|---|---|---|
| clean | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| brightness x0.7 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| brightness x1.3 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| contrast x0.7 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| gaussian blur r=2 | 1.0000 | 0.9915 | 0.9957 | 0 | 1 |
| gaussian noise std=10 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| jpeg quality 30 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| rotation 45 deg | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| zoom (center crop 90%) | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |

## Stress test (severity sweeps)

| perturbation | level | recall | specificity | F1 | FN | FP | quality-gate flagged | FN not flagged |
|---|---|---|---|---|---|---|---|---|
| gaussian_blur_radius | 1 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 14% | 0 |
| gaussian_blur_radius | 2 | 0.9915 | 1.0000 | 0.9957 | 1 | 0 | 99% | 0 |
| gaussian_blur_radius | 3 | 0.9831 | 1.0000 | 0.9915 | 2 | 0 | 100% | 0 |
| gaussian_blur_radius | 4 | 0.9576 | 1.0000 | 0.9784 | 5 | 0 | 100% | 0 |
| gaussian_blur_radius | 6 | 0.8814 | 1.0000 | 0.9369 | 14 | 0 | 100% | 0 |
| brightness_factor | 0.4 | 0.9831 | 1.0000 | 0.9915 | 2 | 0 | 100% | 0 |
| brightness_factor | 0.6 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 100% | 0 |
| brightness_factor | 0.8 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 22% | 0 |
| brightness_factor | 1.2 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 19% | 0 |
| brightness_factor | 1.4 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 83% | 0 |
| brightness_factor | 1.6 | 0.9915 | 1.0000 | 0.9957 | 1 | 0 | 98% | 0 |
| gaussian_noise_std | 5 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| gaussian_noise_std | 10 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| gaussian_noise_std | 20 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| gaussian_noise_std | 30 | 0.9831 | 1.0000 | 0.9915 | 2 | 0 | 0% | 2 |
| gaussian_noise_std | 50 | 0.8814 | 0.9487 | 0.9204 | 14 | 4 | 0% | 14 |
| jpeg_quality | 50 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| jpeg_quality | 20 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| jpeg_quality | 10 | 0.9831 | 1.0000 | 0.9915 | 2 | 0 | 0% | 2 |
| jpeg_quality | 5 | 0.9576 | 1.0000 | 0.9784 | 5 | 0 | 0% | 5 |
| camera_resolution_px | 256 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 35% | 0 |
| camera_resolution_px | 192 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 76% | 0 |
| camera_resolution_px | 128 | 0.9915 | 1.0000 | 0.9957 | 1 | 0 | 100% | 0 |
| camera_resolution_px | 96 | 0.9915 | 1.0000 | 0.9957 | 1 | 0 | 100% | 0 |
| camera_resolution_px | 64 | 0.9322 | 0.9872 | 0.9607 | 8 | 1 | 100% | 0 |

## Hardest test images

- `data/raw/def_front/cast_def_0_1139.jpeg` (defective) P(def)=0.9059
- `data/raw/def_front/cast_def_0_1729.jpeg` (defective) P(def)=0.9438
- `data/raw/def_front/cast_def_0_3460.jpeg` (defective) P(def)=0.9442
- `data/raw/ok_front/cast_ok_0_6055.jpeg` (normal) P(def)=0.0358
- `data/raw/ok_front/cast_ok_0_7342.jpeg` (normal) P(def)=0.0296
- `data/raw/ok_front/cast_ok_0_9840.jpeg` (normal) P(def)=0.0287
