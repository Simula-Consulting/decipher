import itertools

import numpy as np
import pytest
from hypothesis import given, note
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import matthews_corrcoef

from matfact.model.predict.classification_tree import (
    Init,
    SegmentedClassificationTree,
    estimate_probability_thresholds_segment,
)


@st.composite
def probabilities(draw, min_samples=1, max_samples=10, min_states=2, max_states=5):
    return draw(
        arrays(
            float,
            st.tuples(
                st.integers(min_value=min_samples, max_value=max_samples),
                st.integers(min_value=min_states, max_value=max_states),
            ),
            elements=st.floats(min_value=0, max_value=1),
        )
        .filter(lambda p: np.all(np.any(p, axis=1)))  # Finite probabilities
        .map(lambda p: p / np.sum(p, axis=1)[:, None])  # Normalize probabilities
    )


@st.composite
def probability_and_correct(
    draw, min_samples=1, max_samples=10, min_states=2, max_states=5
):
    probability_array = draw(
        probabilities(
            min_samples=min_samples,
            max_samples=max_samples,
            min_states=min_states,
            max_states=max_states,
        )
    )
    number_of_samples, number_of_states = probability_array.shape
    fake_correct = draw(
        arrays(
            int,
            number_of_samples,
            elements=st.integers(min_value=1, max_value=number_of_states),
        )
    )
    return probability_array, fake_correct


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
    clf = SegmentedClassificationTree(thresholds)
    predicted_state = clf.predict(_probabilities, age_segments)
    assert len(predicted_state) == len(age_segments)
    assert np.all(predicted_state == correct_class)


@pytest.mark.parametrize("init_method", [Init.DEFAULT, Init.PARTITION])
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
    init_method: Init,
):
    """Test ClassificationTree estimation."""
    clf = estimate_probability_thresholds_segment(
        np.array(correct_classes),
        np.array(probabilities),
        age_segments,
        number_of_age_segments,
        init_method=init_method,
    )
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


# The optimization objective, Matthew's correlation coefficient, is not strictly larger
# for better predictions.
# Consider, for example, the case where the correct states are
# [1, 1, 1]
# Then, any prediction will give a score of 0.0 (out of -1 to 1)
# >>> for predict in itertools.product(range(3), repeat=3):
# >>>     assert matthews_corrcoef([1,1,1], predict) == 0.0
# Therefore, we cannot optimize the thresholds for these cases.
@pytest.mark.skip(
    reason=(
        "The score function, MCC, is not strictly larger "
        "for correct prediction in all cases."
    )
)
@given(probabilities(min_samples=10, max_samples=12))
def test_classification_predictions_argmax(probabilities):
    """Test that estimation works in the simple case were we set argmax as correct."""
    correct = np.argmax(probabilities, axis=1) + 1  # Compensate for 1-indexed
    number_of_samples = probabilities.shape[0]
    age_segments = [0] * number_of_samples
    clf = estimate_probability_thresholds_segment(
        correct, probabilities, age_segments, 1
    )
    note(f"Correct: {correct}")
    note(f"Thresholds: {clf.thresholds}")
    predicted = clf.predict(probabilities, age_segments)
    note(f"MCC: {matthews_corrcoef(correct, predicted)}")
    assert np.all(predicted == correct)


@pytest.mark.skip(reason="The current solver does not guarantee an optimal solution")
@given(probability_and_correct(min_samples=5, max_states=3))
def test_all_partitions(probability_correct_pair):
    """Test threshold estimation against exhaustive search of all possibilities."""
    probabilities, correct = probability_correct_pair
    age_segments = [0] * probabilities.shape[0]

    # Find optimal score
    threshold_limits = [(*set(p), 1) for p in probabilities.T[1:]]
    clf = SegmentedClassificationTree()
    best_score = -float("inf")
    best_scoring_thresholds = []
    for thresholds in itertools.product(*threshold_limits):
        clf.set_params(thresholds=[thresholds])  # Only one segment, so wrap in [ ]
        prediction = clf.predict(probabilities, age_segments)
        if (score := matthews_corrcoef(correct, prediction)) >= best_score:
            best_score = score
            best_scoring_thresholds.append(thresholds)
        note(f"{thresholds}, {prediction}, {score}")

    # Estimate
    clf_estimated = estimate_probability_thresholds_segment(
        correct, probabilities, age_segments, 1
    )
    prediction_estimated = clf_estimated.predict(probabilities, age_segments)
    score_estimated = matthews_corrcoef(correct, prediction_estimated)
    assert score_estimated == best_score
