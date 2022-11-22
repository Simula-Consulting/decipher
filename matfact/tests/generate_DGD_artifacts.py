"""Script to generate DGD artifacts used in test_data."""
import warnings

import numpy as np
from test_data import DGD_artifact_path, _generate_DGD_data

if __name__ == "__main__":
    warnings.warn(
        "Regenerating DGD test artifacts! "
        "Do not do this unless you are sure, this may invalidate the test."
    )
    dataset = _generate_DGD_data()
    np.save(DGD_artifact_path, dataset.X)
