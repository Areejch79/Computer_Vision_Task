# Error analysis – `mobilenetv3_large_100-20261005-3c50e5a2` on the test split (196 images)

Decision threshold (tuned on validation): **0.2848**

| metric | value | exact (Clopper-Pearson) 95% CI | bootstrap 95% CI |
|---|---|---|---|
| precision | 1.0000 | 0.969 – 1.000 | 1.000 – 1.000 |
| recall | 1.0000 | 0.969 – 1.000 | 1.000 – 1.000 |
| f1 | 1.0000 | – | 1.000 – 1.000 |
| specificity | 1.0000 | 0.954 – 1.000 | 1.000 – 1.000 |
| accuracy | 1.0000 | 0.981 – 1.000 | 1.000 – 1.000 |
| ROC-AUC | 1.0000 | | |
| PR-AUC | 1.0000 | | |

Confusion matrix: TN=78 FP=0 FN=0 TP=118

Expected precision if the line had a lower defect rate (recall/FPR held fixed): 1%: 1.0, 5%: 1.0, 10%: 1.0

## False negatives (missed defects – the costly error)

- none

## False positives (false alarms)

- none

## Mean image statistics by outcome

| outcome   |   p_defective |   border_brightness |   mean_brightness |   contrast |   sharpness |
|:----------|--------------:|--------------------:|------------------:|-----------:|------------:|
| TN        |         0.012 |             194.097 |           149.627 |     62.416 |     141.893 |
| TP        |         0.967 |             168.432 |           139.219 |     60.139 |     164.013 |

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
| gaussian blur r=2 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| gaussian noise std=10 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| jpeg quality 30 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| rotation 45 deg | 0.9916 | 1.0000 | 0.9958 | 1 | 0 |
| zoom (center crop 90%) | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |

## Stress test (severity sweeps)

| perturbation | level | recall | specificity | F1 | FN | FP | quality-gate flagged | FN not flagged |
|---|---|---|---|---|---|---|---|---|
| gaussian_blur_radius | 1 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 14% | 0 |
| gaussian_blur_radius | 2 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 99% | 0 |
| gaussian_blur_radius | 3 | 0.9831 | 1.0000 | 0.9915 | 2 | 0 | 100% | 0 |
| gaussian_blur_radius | 4 | 0.9407 | 1.0000 | 0.9694 | 7 | 0 | 100% | 0 |
| gaussian_blur_radius | 6 | 0.7881 | 0.9872 | 0.8774 | 25 | 1 | 100% | 0 |
| brightness_factor | 0.4 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 100% | 0 |
| brightness_factor | 0.6 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 100% | 0 |
| brightness_factor | 0.8 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 22% | 0 |
| brightness_factor | 1.2 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 19% | 0 |
| brightness_factor | 1.4 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 83% | 0 |
| brightness_factor | 1.6 | 0.9746 | 1.0000 | 0.9871 | 3 | 0 | 98% | 0 |
| gaussian_noise_std | 5 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| gaussian_noise_std | 10 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| gaussian_noise_std | 20 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 100% | 0 |
| gaussian_noise_std | 30 | 0.9831 | 1.0000 | 0.9915 | 2 | 0 | 100% | 0 |
| gaussian_noise_std | 50 | 0.9153 | 1.0000 | 0.9558 | 10 | 0 | 100% | 0 |
| jpeg_quality | 50 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| jpeg_quality | 20 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0% | 0 |
| jpeg_quality | 10 | 0.9746 | 0.9103 | 0.9583 | 3 | 7 | 0% | 3 |
| jpeg_quality | 5 | 0.9831 | 0.2692 | 0.7973 | 2 | 57 | 0% | 2 |
| camera_resolution_px | 256 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 35% | 0 |
| camera_resolution_px | 192 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 76% | 0 |
| camera_resolution_px | 128 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 100% | 0 |
| camera_resolution_px | 96 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 100% | 0 |
| camera_resolution_px | 64 | 0.8898 | 0.9872 | 0.9375 | 13 | 1 | 100% | 0 |

## Hardest test images

- `data/raw/def_front/cast_def_0_177.jpeg` (defective) P(def)=0.9244
- `data/raw/def_front/cast_def_0_4239.jpeg` (defective) P(def)=0.9271
- `data/raw/def_front/cast_def_0_9879.jpeg` (defective) P(def)=0.9348
- `data/raw/ok_front/cast_ok_0_3106.jpeg` (normal) P(def)=0.0400
- `data/raw/ok_front/cast_ok_0_5548.jpeg` (normal) P(def)=0.0225
- `data/raw/ok_front/cast_ok_0_6773.jpeg` (normal) P(def)=0.0215
