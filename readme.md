# Offshore PPE Violation Attribution Model

A machine learning and rule-based attribution system designed to identify which worker committed a Personal Protective Equipment (PPE) violation detected by camera systems in offshore operational zones.

---

## Overview

Camera feeds identify when and where PPE safety violations occur, but multi-worker zones can make attribution ambiguous. This project combines:
1. **XGBoost Classifier**: Predicts candidate violation probabilities based on sensor, movement, and zone features.
2. **Post-Processing Rule Engine**: Applies domain-specific safety heuristics (single-occupancy overrides, dwell-time penalties, and confidence thresholding) to make decisive alerts (`NAMED_ALERT` vs `GROUP_ALERT`).

---

## Project Structure

```text
offshoreModel/
├── data/
│   ├── attribution_labels.csv
│   ├── camera_events.csv
│   ├── crew_roster.csv
│   ├── expanded_candidates.csv
│   ├── health_alerts.csv
│   ├── training_features.csv
│   ├── wearable_events.csv
│   └── zone_mapping.csv
├── models/
│   └── xgboost_model.pkl          # Saved trained model
├── outputs/
│   ├── classification_report.txt  # Test evaluation metrics
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   ├── feature_importance.png     # Feature importance plot
│   └── roc_curve.png              # ROC AUC curve visualization
├── src/
│   ├── config.py                  # Model hyperparameters, features, and paths
│   ├── evaluate.py                # Evaluation pipeline and artifact generation
│   ├── predict.py                 # Candidate ranking & rule-engine inference
│   ├── requirements.txt           # Dependencies
│   ├── rules.py                   # Domain safety heuristic rules
│   └── train.py                   # Model training script
├── .gitignore
├── requirements.txt               # Root dependencies
└── readme.md                      # Project documentation
```

---

## Installation & Requirements

### Prerequisites
- Python 3.10+ (tested on Python 3.12 / 3.14)

### Install Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

Installed packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`
- `matplotlib`

---

## Usage

You can run scripts either from the project root or from inside the `src/` directory.

### 1. Train the Model
Trains the XGBoost classifier on `data/training_features.csv` and saves the model to `models/xgboost_model.pkl`:

```bash
python src/train.py
```

### 2. Evaluate Performance
Evaluates the trained model against the hold-out test set and saves evaluation plots & metrics into `outputs/`:

```bash
python src/evaluate.py
```
Generated artifacts in `outputs/`:
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

### 3. Run Predictions & Rule-Based Attribution
Runs the candidate pool through model inference and safety rule overrides to determine the final decision:

```bash
python src/predict.py
```

---

## Features & Rule Logic

### Model Features
- `delta_t_entry_s`: Entry time delta in seconds
- `dwell_time_s`: Total duration the worker has been in the zone
- `zone_occupancy`: Number of workers present in the zone
- `motion_match`: Binary motion correlation with the camera event
- `path_prior_score`: Historical path trajectory match score
- `is_red_zone`: Flag for hazardous/restricted zones
- `shift_hour`: Hour of the shift during the incident

### Rule Engine Heuristics
- **Single Occupancy Override**: If zone occupancy is 1, confidence is set to 99% (`NAMED_ALERT`).
- **Dwell Time Penalty**: Workers dwelling over 300 seconds (5 minutes) receive a probability penalty (0.8x multiplier).
- **Threshold Decision**: Confidence $\ge 70\%$ issues a `NAMED_ALERT`, otherwise a `GROUP_ALERT`.
