import numpy as np

from hmm_synthetic.plotting import history_panel


def test_sample_histories():
    histories = np.array(
        [
            [3, 1, 2, 4, 1, 3, 1, 1, 1, 2],
            [3, 1, 1, 1, 4, 1, 1, 3, 3, 1],
            [3, 3, 1, 1, 1, 2, 3, 1, 1, 4],
            [2, 4, 2, 2, 1, 1, 1, 2, 2, 1],
            [1, 3, 1, 1, 3, 1, 2, 1, 1, 1],
            [3, 1, 2, 3, 3, 1, 3, 3, 1, 1],
            [1, 2, 1, 1, 1, 2, 1, 1, 1, 1],
            [2, 1, 1, 3, 1, 2, 2, 1, 1, 1],
            [1, 1, 4, 1, 3, 1, 1, 2, 3, 4],
            [1, 1, 1, 2, 4, 3, 2, 2, 1, 4],
        ]
    )
    _, time_steps = histories.shape
    number_to_sample = 3  # Chosen arbitrarily

    sampled_histories_list = [
        history_panel.sample_histories(
            histories,
            size=number_to_sample,
            rnd=np.random.default_rng(),
            risk_stratify=risk_stratify,
        )
        for risk_stratify in [True, False]
    ]
    for samples in sampled_histories_list:
        assert samples.shape == (number_to_sample, time_steps)
    assert np.any(sampled_histories_list[0] != sampled_histories_list[1])
