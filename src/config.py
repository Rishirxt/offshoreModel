FEATURES = [
    "delta_t_entry_s",
    "dwell_time_s",
    "zone_occupancy",
    "motion_match",
    "path_prior_score",
    "is_red_zone",
    "shift_hour"
]

TARGET = "is_violator"

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = str(BASE_DIR / "data" / "training_features.csv")
MODEL_PATH = str(BASE_DIR / "models" / "xgboost_model.pkl")
OUTPUTS_DIR = str(BASE_DIR / "outputs")
EXPANDED_CANDIDATES_PATH = str(BASE_DIR / "data" / "expanded_candidates.csv")

# Ensure required directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

XGB_PARAMS = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "random_state": RANDOM_STATE,
    "scale_pos_weight": 8.6,
    "eval_metric": "auc"
}


