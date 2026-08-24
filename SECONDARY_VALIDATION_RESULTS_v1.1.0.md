# Secondary real-data validation results — v1.1.0

These two datasets were prospectively specified in `SECONDARY_VALIDATION_PROTOCOL_v1.1.0.md` before predictive performance was inspected. The core `plabn.py` is byte-identical to manuscript-lock v1.0.0 (SHA-256 c8e0a286213a87ecaf5a9ab03e473d76989c92ab5856520c60993a98bfb40751).

## 1. NHANES 2015–2016 blood-pressure definition experiment

Analytic sample: 5,012 adults age >=20 with >=2 valid paired auscultatory readings and required non-BP predictors.

Canonical three-level within-visit BP screening phenotype:
- class 0: mean SBP <130 and mean DBP <80;
- class 1: >=130/80 but below 140/90;
- class 2: mean SBP >=140 or mean DBP >=90.

Observed definitions:
- ACC/AHA-style threshold: positive at >=130/80;
- ESC/conventional threshold: positive at >=140/90.

The deterministic stacked operators have rank 3 and condition number 2.4142. All 25 PLA-BN fits reached tolerance convergence.

### Repeated 5x5 canonical performance

| Method | Accuracy mean (SD) | Balanced accuracy mean (SD) | Macro-F1 mean (SD) | Log-loss mean (SD) | Brier mean (SD) |
|---|---:|---:|---:|---:|---:|
| GALIATSATOS/PLA-BN TAN | 0.5700 (0.0117) | 0.4013 (0.0117) | **0.3995 (0.0138)** | 0.9855 (0.0455) | 0.5514 (0.0101) |
| Operator-aware logistic EM | 0.5947 (0.0116) | 0.4025 (0.0127) | 0.3873 (0.0179) | 0.8732 (0.0117) | 0.5113 (0.0072) |
| Oracle-label logistic | **0.6000 (0.0084)** | 0.3998 (0.0098) | 0.3667 (0.0115) | **0.8629 (0.0087)** | **0.5078 (0.0059)** |

Corrected comparisons:
- PLA-BN vs operator-aware logistic: accuracy difference -0.0247, Holm-adjusted corrected p=0.0181; macro-F1 difference +0.0122, adjusted p=0.8160.
- PLA-BN vs oracle logistic: accuracy difference -0.0300, adjusted p=0.00205; macro-F1 difference **+0.0328**, corrected 95% CI **0.0148 to 0.0508**, Holm-adjusted corrected p=**0.00484**.
- Calibration/probability scores remain significantly worse for PLA-BN than both logistic comparators.

### Definition transport

| Definition | Method | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| >=130/80 | PLA-BN | 0.6404 | 0.6192 | 0.6201 |
| >=130/80 | Operator-aware logistic EM | 0.6590 | 0.6313 | 0.6331 |
| >=140/90 | PLA-BN | 0.7906 | **0.5415** | **0.5393** |
| >=140/90 | Operator-aware logistic EM | 0.8067 | 0.5268 | 0.5070 |

Interpretation: hypertension is a stronger main-text validation than the diabetes benchmark because the alternative definitions operate on the same underlying BP measurement scale and the deterministic operator system is exactly full-rank. PLA-BN does not maximize raw accuracy or calibration, but it shows a favorable class-balanced trade-off, including significantly higher macro-F1 than the oracle logistic comparator in the prespecified canonical task.

## 2. UCI White Wine Quality experiment

All 4,898 white-wine records were retained. There are 3,961 unique predictor vectors; exact predictor duplicates were group-protected so no duplicate predictor vector crossed an outer train/test split.

Canonical outcome:
- class 0: quality <=5 (n=1,640);
- class 1: quality =6 (n=2,198);
- class 2: quality >=7 (n=1,060).

Observed definitions were quality >=6 and quality >=7. The deterministic stacked operators have rank 3 and condition number 2.4142. All 25 PLA-BN fits reached tolerance convergence and every outer split had zero duplicate-group overlap.

### Repeated 5x5 canonical performance

| Method | Accuracy mean (SD) | Balanced accuracy mean (SD) | Macro-F1 mean (SD) | Log-loss mean (SD) | Brier mean (SD) |
|---|---:|---:|---:|---:|---:|
| GALIATSATOS/PLA-BN TAN | 0.5448 (0.0187) | **0.5465 (0.0199)** | 0.5406 (0.0187) | 1.1273 (0.0694) | 0.6292 (0.0280) |
| Operator-aware logistic EM | 0.5710 (0.0173) | 0.5390 (0.0197) | 0.5508 (0.0196) | 0.8864 (0.0218) | 0.5392 (0.0129) |
| Oracle-label logistic | **0.5755 (0.0158)** | 0.5365 (0.0158) | **0.5493 (0.0165)** | **0.8825 (0.0211)** | **0.5377 (0.0130)** |

Corrected comparisons:
- PLA-BN vs operator-aware logistic accuracy: -0.0262, Holm-adjusted corrected p=0.0495.
- PLA-BN vs oracle logistic accuracy: -0.0307, adjusted p=0.00632.
- Balanced-accuracy and macro-F1 differences versus both comparators are not significant after Holm correction.
- PLA-BN log-loss and Brier score are significantly worse.

### Definition transport

| Definition | Method | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| quality >=6 | PLA-BN | 0.7518 | **0.7159** | **0.7181** |
| quality >=6 | Operator-aware logistic EM | 0.7550 | 0.6932 | 0.7035 |
| quality >=7 | PLA-BN | 0.7688 | **0.6690** | **0.6656** |
| quality >=7 | Operator-aware logistic EM | 0.8040 | 0.6245 | 0.6429 |

Interpretation: Wine Quality provides useful non-clinical confirmation. PLA-BN sacrifices some raw accuracy and calibration but has competitive canonical class-balanced performance and stronger balanced accuracy/macro-F1 after transport to both binary thresholds.

## Recommended manuscript placement

Main text:
1. Controlled simulation (regenerated with frozen core before submission).
2. NHANES Hypertension as the principal natural-threshold real-data demonstration.
3. NHANES Diabetes as a harder natural-definition discordance experiment.
4. UCI Wine Quality as cross-domain validation.

Supplementary:
- Heart Disease external-population stress test.
- Detailed classwise and transport tables.
- Sensitivity analyses and all fold-level audit files.

No result from either v1.1.0 dataset was discarded after performance inspection.
