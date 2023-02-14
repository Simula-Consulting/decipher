"""Script to generate SCMF artifacts used in test_scmf."""
import warnings

import numpy as np
from test_scmf import _generate_SCMF_logs

from decipher.matfact.settings import settings

artifact_path = settings.paths.test / "test_artifacts" / "SCMF_test"

if __name__ == "__main__":
    warnings.warn(
        "Regenerating SCMF test artifacts! "
        "Do not do this unless you are sure, this may invalidate the test."
    )
    logs = _generate_SCMF_logs()
    for attribute in logs:
        np.save(artifact_path / f"{attribute}_log.npy", logs[attribute])
