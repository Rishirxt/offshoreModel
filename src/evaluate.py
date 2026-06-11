import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

from xgboost import plot_importance

from config import (
    FEATURES,
    TARGET,
    DATA_PATH,
    MODEL_PATH,
    OUTPUTS_DIR,
    TEST_SIZE,
    RANDOM_STATE
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]

# Split to only evaluate on the unseen test set
_, X, _, y = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully")

# --------------------------------------------------
# Predict Probabilities
# --------------------------------------------------

y_prob = model.predict_proba(X)[:, 1]

# 70% Threshold from Design Document
y_pred = (y_prob >= 0.70).astype(int)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------

report = classification_report(
    y,
    y_pred
)

print("\nClassification Report\n")
print(report)

with open(
    f"{OUTPUTS_DIR}/classification_report.txt",
    "w"
) as f:
    f.write(report)

print("Saved: classification_report.txt")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig(
    f"{OUTPUTS_DIR}/confusion_matrix.png"
)

plt.close()

print("Saved: confusion_matrix.png")

# --------------------------------------------------
# ROC AUC
# --------------------------------------------------

auc = roc_auc_score(
    y,
    y_prob
)

print(f"\nROC AUC: {auc:.4f}")

# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

fpr, tpr, _ = roc_curve(
    y,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUTS_DIR}/roc_curve.png"
)

plt.close()

print("Saved: roc_curve.png")

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plot_importance(
    model,
    max_num_features=10
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    f"{OUTPUTS_DIR}/feature_importance.png"
)

plt.close()

print("Saved: feature_importance.png")

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n=================================")
print("Evaluation Complete")
print("=================================")
print(f"ROC AUC: {auc:.4f}")
print(f"Outputs saved in {OUTPUTS_DIR}")