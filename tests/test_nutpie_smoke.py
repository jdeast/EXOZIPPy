"""nutpie actually samples on this platform.

examples/hat3 -- the README's install check -- uses `method: nutpie`
because it needs no jax and so is the fastest sampler that also runs on Intel
Macs. Nothing else in the suite samples with nutpie, so without this the
Intel-Mac CI job (which runs this file in its smoke subset) would install
nutpie and never once prove its numba-compiled path works there.
"""

import numpy as np
import pymc as pm
import pytest


def test_nutpie_samples_a_toy_model():
    """
    Given a one-parameter Normal model,
    When it is sampled with nuts_sampler="nutpie",
    Then the draws recover the mean and come back finite.
    """
    # Arrange
    pytest.importorskip("nutpie")
    with pm.Model():
        pm.Normal("x", mu=3.0, sigma=1.0)

        # Act
        idata = pm.sample(
            draws=200,
            tune=200,
            chains=2,
            nuts_sampler="nutpie",
            random_seed=1,
            progressbar=False,
        )

    # Assert
    x = idata.posterior["x"].values
    assert x.shape == (2, 200)
    assert np.all(np.isfinite(x))
    assert abs(x.mean() - 3.0) < 0.3
