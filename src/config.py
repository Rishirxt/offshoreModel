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

DATA_PATH = "../data/training_features.csv"
MODEL_PATH = "../models/xgboost_model.pkl"
OUTPUTS_DIR = "../outputs"

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


