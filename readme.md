# Offshore PPE Violation Attribution System

## Overview

This project implements an ML-powered attribution engine for offshore platform safety monitoring.

The system combines wearable device telemetry and camera-based PPE violation detection to determine which worker most likely caused a detected safety violation.

The camera system identifies that a PPE violation has occurred, while the wearable system identifies who is present in a given zone. The attribution engine bridges these two systems by using machine learning to estimate which worker is responsible for the violation.

---

## Business Problem

The offshore platform generates two independent data streams:

### Wearable Gateway

Provides:

* Worker identity
* Zone location
* Health telemetry
* Motion information

Cannot provide:

* PPE compliance status

### Camera Gateway

Provides:

* PPE violation detection
* Camera zone
* Evidence image
* Detection confidence

Cannot provide:

* Worker identity

The challenge is:

> Given a camera violation event and multiple workers present in the same zone, determine which worker most likely caused the violation.

---

## Project Objective

Build an attribution model that predicts:

```python
is_violator
```

Where:

* 1 = Candidate worker caused the PPE violation
* 0 = Candidate worker did not cause the PPE violation

The model outputs a probability score for every candidate worker.

The highest scoring candidate is selected for alert routing.

---

## Dataset

### Training Dataset

`training_features.csv`

Rows: 1439

Target:

```python
is_violator
```

Class Distribution:

* Positive: 150
* Negative: 1289

Approximate ratio:

```text
1 : 8.6
```

---

## Features Used

| Feature          | Description                                  |
| ---------------- | -------------------------------------------- |
| delta_t_entry_s  | Time between zone entry and camera violation |
| dwell_time_s     | Time spent in zone before violation          |
| zone_occupancy   | Number of workers present in zone            |
| motion_match     | Accelerometer movement match                 |
| path_prior_score | Route plausibility score                     |
| is_red_zone      | High-risk area indicator                     |
| shift_hour       | Hour within worker shift                     |

---

## Machine Learning Model

Algorithm:

```text
XGBoost Binary Classification
```

Configuration:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=8.6,
    eval_metric="auc"
)
```

---

## System Architecture

```text
Camera Violation
        │
        ▼
Candidate Worker Pool
        │
        ▼
Feature Extraction
        │
        ▼
XGBoost Model
        │
        ▼
Probability Score
        │
        ▼
Business Rules
        │
        ▼
Named Alert / Group Alert
```

---

## Business Rules

### Rule 1 — Single Occupancy Override

If:

```text
zone_occupancy == 1
```

Then:

```text
Confidence = 99%
```

Named alert is issued immediately.

---

### Rule 2 — Dwell Time Penalty

If:

```text
dwell_time_s > 300
```

Apply confidence reduction.

Workers with long dwell times are less likely to be responsible for a newly detected violation.

---

### Rule 3 — Alert Threshold

If:

```text
Probability >= 70%
```

Issue:

```text
Named Alert
```

Else:

```text
Group Alert
```

---

## Model Performance

Evaluation Metrics:

```text
Precision : 0.78
Recall    : 0.61
F1 Score  : 0.69
ROC-AUC   : 0.9451
```

Confusion Matrix:

```text
[[1263   26]
 [  58   92]]
```

---

## Project Structure

```text
OffshoreModel/
│
├── data/
│
├── models/
│   └── xgboost_model.pkl
│
├── outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── src/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── rules.py
│
└── README.md
```

---

## Future Enhancements

* Hyperparameter optimization
* Cross-validation
* SHAP explainability
* Real-time MQTT integration
* Automatic feature generation from raw wearable and camera streams
* Production deployment on edge infrastructure
* Supervisor feedback loop for continuous learning

---

## Conclusion

This project demonstrates an end-to-end PPE violation attribution pipeline using XGBoost and rule-based decision logic. The system enables targeted safety alerts by identifying the most likely worker responsible for a detected violation while minimizing false attribution through confidence thresholds and business rules.
