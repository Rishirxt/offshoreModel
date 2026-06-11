import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
'''
The model predicts:
Which user committed the PPE violation that the camera already detected.
'''
from config import (
    FEATURES,
    TARGET,
    DATA_PATH,
    MODEL_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    XGB_PARAMS
)

# --------------------------
# Load Dataset
# --------------------------

df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape}")

# --------------------------
# Features / Target
# --------------------------

X = df[FEATURES]
y = df[TARGET]

# --------------------------
# Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Training Rows: {len(X_train)}")
print(f"Testing Rows: {len(X_test)}")

# --------------------------
# Model
# --------------------------

model = XGBClassifier(**XGB_PARAMS)

# --------------------------
# Train
# --------------------------

model.fit(
    X_train,
    y_train
)

# --------------------------
# Evaluate
# --------------------------

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(
    y_test, 
    y_prob
)

print(f"\nROC AUC: {auc:.4f}")

# --------------------------
# Save Model
# --------------------------

joblib.dump(
    model,
    MODEL_PATH
)

print(f"\nModel Saved -> {MODEL_PATH}")



