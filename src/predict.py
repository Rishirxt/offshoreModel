import joblib
import pandas as pd

from config import (
    FEATURES,
    MODEL_PATH
)

from rules import apply_rules


# ------------------------------------
# Load Model
# ------------------------------------

model = joblib.load(MODEL_PATH)

print("Model Loaded\n")


# ------------------------------------
# Example Candidate Pool
# ------------------------------------

candidates = pd.read_csv("../data/expanded_candidates.csv")

# ------------------------------------
# Model Prediction
# ------------------------------------

X = candidates[FEATURES]

candidates["probability"] = (
    model.predict_proba(X)[:, 1]
)
results = []
for _, row in candidates.iterrows():
    candidate = row.to_dict()
    result = apply_rules(candidate)
    results.append({
        "candidate_id": candidate["candidate_id"],
        "final_probability": result["probability"],
        "decision": result["decision"],
        "reason": result["reason"],
        "motion_match": candidate["motion_match"],
        "dwell_time_s": candidate["dwell_time_s"]
    })
results_df = pd.DataFrame(results)

# ------------------------------------
# Rank Candidates
# ------------------------------------

results_df = results_df.sort_values(
    by=["final_probability", "motion_match", "dwell_time_s"],
    ascending=[False, False, True]
)

print("\nCandidate Ranking\n")

print(results_df)

# ------------------------------------
# Winner
# ------------------------------------

winner = results_df.iloc[0]

print("\n--------------------------------")
print("FINAL DECISION")
print("--------------------------------")

print(
    f"Candidate : {winner['candidate_id']}"
)

print(
    f"Probability : {winner['final_probability']:.2%}"
)
print(
    f"Decision : {winner['decision']}"
)
print(
    f"Reason : {winner['reason']}"
)