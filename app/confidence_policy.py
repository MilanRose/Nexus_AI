
def estimate_confidence(sample_size: int) -> float:
    """
    Return a provisional sample-sufficiency score.

    This is an engineering heuristic, NOT a calibrated probability
    that a finding is true. It considers sample size only.

    Provisional thresholds:
        0 observations:       0.00
        1-9 observations:     0.50
        10-29 observations:  0.65
        30-99 observations:  0.80
        100+ observations:   0.90
    """
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size < 0
    ):
        raise ValueError("sample_size must be a non-negative integer.")

    if sample_size == 0:
        return 0.0
    if sample_size < 10:
        return 0.50
    if sample_size < 30:
        return 0.65
    if sample_size < 100:
        return 0.80

    return 0.90
