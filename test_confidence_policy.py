
import pytest

from app.confidence_policy import estimate_confidence


@pytest.mark.parametrize(
    ("sample_size", "expected"),
    [
        (0, 0.0),
        (1, 0.50),
        (9, 0.50),
        (10, 0.65),
        (29, 0.65),
        (30, 0.80),
        (99, 0.80),
        (100, 0.90),
        (891, 0.90),
    ],
)
def test_estimate_confidence_thresholds(sample_size, expected):
    assert estimate_confidence(sample_size) == expected


@pytest.mark.parametrize(
    "sample_size",
    [-1, True, False, 1.5, "30", None],
)
def test_estimate_confidence_rejects_invalid_sample_sizes(sample_size):
    with pytest.raises(ValueError):
        estimate_confidence(sample_size)
