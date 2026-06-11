# -----------------------------------------
# Rule 1
# Single Occupancy Override
# -----------------------------------------

def single_occupancy_override(zone_occupancy):
    """
    If only one worker is present in the zone,
    attribution is essentially certain.
    """

    if zone_occupancy == 1:
        return True

    return False


# -----------------------------------------
# Rule 2
# Dwell Time Penalty
# -----------------------------------------

def apply_dwell_penalty(probability, dwell_time_s):
    """
    Workers who have been in the zone for a long time
    are less likely to be the violator.

    PPT threshold:
    300 seconds = 5 minutes
    """

    if dwell_time_s > 300:
        probability *= 0.8

    return probability


# -----------------------------------------
# Rule 3
# Alert Decision
# -----------------------------------------

def threshold_decision(probability):
    """
    70% threshold from design document.
    """

    if probability >= 0.70:
        return "NAMED_ALERT"

    return "GROUP_ALERT"


# -----------------------------------------
# Full Rule Engine
# -----------------------------------------

def apply_rules(candidate):
    """
    candidate should contain:

    probability
    zone_occupancy
    dwell_time_s
    """

    # Rule 1
    if single_occupancy_override(
        candidate["zone_occupancy"]
    ):
        return {
            "probability": 0.99,
            "decision": "NAMED_ALERT",
            "reason": "Single Occupancy Override"
        }

    # Rule 2
    adjusted_probability = apply_dwell_penalty(
        candidate["probability"],
        candidate["dwell_time_s"]
    )

    # Rule 3
    decision = threshold_decision(
        adjusted_probability
    )

    return {
        "probability": adjusted_probability,
        "decision": decision,
        "reason": "Model + Rules"
    }