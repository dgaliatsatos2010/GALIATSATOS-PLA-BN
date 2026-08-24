# Manuscript Results — GALIATSATOS / PLA-BN v1.0.0

## Primary repeated 5x5 NHANES validation

Dataset: NHANES 2015–2016 adults with complete HbA1c, fasting plasma glucose (FPG), and 2-h OGTT data.  
Participants: **1,631**.  
Outer validation: **5 repeats × 5 folds = 25 outer evaluations**.

### Canonical screening-severity performance

| Method | Accuracy mean (SD) | Balanced accuracy mean (SD) | Macro-F1 mean (SD) | Log-loss mean (SD) | Brier mean (SD) |
|---|---:|---:|---:|---:|---:|
| GALIATSATOS/PLA-BN TAN | 0.5517 (0.0352) | 0.4208 (0.0268) | 0.4189 (0.0274) | 1.2060 (0.0763) | 0.6312 (0.0349) |
| Operator-aware logistic EM | 0.6262 (0.0245) | 0.4543 (0.0185) | 0.4383 (0.0218) | 0.8449 (0.0332) | 0.5091 (0.0199) |
| Oracle-label logistic | 0.6482 (0.0206) | 0.4524 (0.0178) | 0.4349 (0.0184) | 0.8073 (0.0227) | 0.4862 (0.0128) |

### Repeat-level 95% confidence intervals

For GALIATSATOS/PLA-BN TAN:

- accuracy: **0.5517**, 95% CI **0.5279–0.5755**;
- balanced accuracy: **0.4208**, 95% CI **0.4068–0.4347**;
- macro-F1: **0.4189**, 95% CI **0.4058–0.4320**;
- log-loss: **1.2060**, 95% CI **1.1709–1.2411**;
- Brier: **0.6312**, 95% CI **0.6176–0.6447**.

### Correlated repeated-CV comparisons

PLA-BN versus operator-aware logistic EM:

- accuracy difference: **−0.0746**; corrected 95% CI **−0.1081 to −0.0410**; Holm-adjusted corrected p = **0.000588**;
- balanced-accuracy difference: **−0.0335**; Holm-adjusted corrected p = **0.1175**;
- macro-F1 difference: **−0.0194**; corrected 95% CI **−0.0517 to 0.0129**; Holm-adjusted corrected p = **0.4558**;
- log-loss difference: **+0.3612** (higher is worse); Holm-adjusted corrected p < **0.000001**;
- Brier difference: **+0.1221** (higher is worse); Holm-adjusted corrected p < **0.000001**.

PLA-BN versus oracle-label logistic:

- accuracy difference: **−0.0965**; corrected 95% CI **−0.1381 to −0.0549**; Holm-adjusted corrected p = **0.000425**;
- balanced-accuracy difference: **−0.0316**; corrected interval includes zero after correction context; Holm-adjusted corrected p = **0.1517**;
- macro-F1 difference: **−0.0159**; corrected 95% CI **−0.0481 to 0.0162**; Holm-adjusted corrected p = **0.4558**;
- log-loss difference: **+0.3987**; Holm-adjusted corrected p < **0.000001**;
- Brier difference: **+0.1450**; Holm-adjusted corrected p < **0.000001**.

The exact Wilcoxon test on only five repeat-level means yields p=0.0625 for these consistently signed differences before multiplicity correction; this reflects the very low resolution/power of an exact paired test with n=5 and is not interpreted as evidence of equivalence.

### Classwise audit from participant-averaged repeated OOF probabilities

For canonical class 2 (diabetes-range screening severity):

- GALIATSATOS/PLA-BN TAN: precision **0.0952**, recall **0.0242**, F1 **0.0386**;
- operator-aware logistic EM: precision/recall/F1 **0/0/0**;
- oracle-label logistic: precision/recall/F1 **0/0/0**.

Thus the proposed method identifies a small subset of the rare severe class, but performance remains poor and must be presented as a limitation, not a clinical claim.

### Transport performance

Mean transported-definition accuracy for GALIATSATOS/PLA-BN TAN:

- HbA1c: **0.6462**;
- FPG: **0.5755**;
- OGTT: **0.7224**.

Corresponding operator-aware logistic EM accuracies are 0.6594, 0.6195, and 0.7382.

## Optimization and identifiability audit

- PLA-BN tolerance convergence: **25/25** outer fits.
- Operator stack full column rank: **25/25** outer fits.
- Operator condition-number range: **2.3515–3.1840**.
- Outer-test data used for operator estimation: **never**.
- Outer-test data used to fit preprocessing: **never**.
- Canonical labels supplied to proposed fit: **never**.

## Manuscript interpretation

The repeated validation does **not** support a claim that PLA-BN is the most accurate or best-calibrated classifier. Operator-aware and oracle logistic models have higher overall accuracy and substantially better probability calibration.

The defensible methodological contribution is instead the ability to learn a canonical posterior representation and transport it across heterogeneous observed outcome definitions under an explicit rank gate, without supplying canonical labels to the proposed estimator. In the primary clinical experiment, macro-F1 and balanced-accuracy differences are smaller than the accuracy/calibration gaps, but non-significance must not be called equivalence.
