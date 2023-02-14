import numpy as np
import pytest

from decipher.matfact.model.matfact import ClassificationTreePredictor
from decipher.matfact.model.predict.classification_tree import (
    ClassificationTree,
    ThresholdInitMethod,
    estimate_probability_thresholds,
)


@pytest.mark.parametrize(
    "segments, time_points, correct_segment_indices",
    [
        [[76, 116], [75, 76, 77, 115, 116, 117], [0, 0, 1, 1, 1, 2]],
        [[116], [75, 76, 77, 115, 116, 117], [0, 0, 0, 0, 0, 1]],
        [[], [75, 76, 77, 115, 116, 117], [0, 0, 0, 0, 0, 0]],
    ],
)
def test_age_segment_index(
    segments: list[int], time_points: list[int], correct_segment_indices: list[int]
):
    """Test that _age_segment_index returns the correct age segment."""
    predictor = ClassificationTreePredictor(segments=segments)
    assert (
        predictor._age_segment_index(np.array(time_points)) == correct_segment_indices
    )


@pytest.mark.parametrize(
    "probabilities, age_segments, thresholds, correct_class",
    # NB! Classes are 1-indexed.
    (
        # Single sample
        (([0.9, 0.1],), (0,), [[0.1], [0.2]], (2,)),
        (([0.9, 0.1],), (1,), [[0.1], [0.2]], (1,)),
        (([0.7, 0.2, 0.1],), (0,), [[0.1, 0.1], [0.15, 0.12]], (3,)),
        (([0.7, 0.2, 0.1],), (1,), [[0.1, 0.1], [0.15, 0.12]], (2,)),
        # Multiple samples
        (([0.9, 0.1], [0.9, 0.1]), (0, 1), [[0.1], [0.2]], (2, 1)),
    ),
)
def test_segmented_classification_tree_prediction(
    probabilities: list[list[float]],
    age_segments: list[int],
    thresholds: list[list[float]],
    correct_class: list[int],
):
    """Test SegmentedClassificationTree predictions."""
    _probabilities = np.array(probabilities)
    clf = ClassificationTree(thresholds)
    predicted_state = clf.predict(_probabilities, age_segments)
    assert len(predicted_state) == len(age_segments)
    assert np.all(predicted_state == correct_class)


@pytest.mark.parametrize(
    "init_method", [ThresholdInitMethod.DEFAULT, ThresholdInitMethod.PARTITION]
)
@pytest.mark.parametrize(
    "probabilities,correct_classes,age_segments,number_of_age_segments,max_thresholds",
    [
        [
            [[0, 1], [1, 0], [0, 1]],
            [2, 1, 2],
            [0, 0, 0],
            1,
            [[1 - np.finfo("float").eps]],
        ],
        [[[0.9, 0.1], [1, 0], [0.8, 0.2]], [2, 1, 2], [0, 0, 0], 1, [[0.1]]],
        [[[0.9, 0.1], [1, 0], [0.8, 0.2]], [2, 1, 2], [1, 0, 0], 2, [[0.2], [0.1]]],
    ],
)
def test_segmented_classification_tree_estimation(
    probabilities: list[list[float]],
    correct_classes: list[int],
    age_segments: list[int],
    number_of_age_segments: int,
    max_thresholds: list[list[float]],
    init_method: ThresholdInitMethod,
):
    """Test ClassificationTree estimation.

    The solution of the thresholds are degenerate: in simple constructed cases, it
    just has to be smaller than the value that would result in the 'next' class
    to be predicted instead. The max_thresholds gives the highest accepted value.
    """

    threholds = estimate_probability_thresholds(
        np.array(correct_classes),
        np.array(probabilities),
        age_segments,
        number_of_age_segments,
        init_method=init_method,
    )
    clf = ClassificationTree(threholds)
    for age_index in range(number_of_age_segments):
        assert all(
            max_threshold >= threshold
            for max_threshold, threshold in zip(
                max_thresholds[age_index], clf.thresholds[age_index]
            )
        ), (
            f"Thresholds too big. Got {clf.thresholds}, required max {max_thresholds}"
            f"Predicted {clf.predict(np.array(probabilities), age_segments)}."
        )
