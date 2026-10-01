"""The posterior a gradient-free sampler builds (PTDE, ptde_async, nested --
all through ``_common.assemble_inference_data``) keeps every variable's own
model shape, like the PyMC-built traces do.

A one-element vector (a lone star's ``teffsed``) used to be squeezed to a
scalar, so ``examples/hat3``'s PTDE trace disagreed in shape with its numpyro
one, and NextGen's SED plot indexed a scalar per star and failed on every
posterior draw.
"""

import logging

import numpy as np

from exozippy.samplers._common import assemble_inference_data


def test_a_one_element_vector_keeps_its_dim_and_a_scalar_stays_scalar():
    """
    Given stored raw draws for a length-1 vector and a true scalar,
    When the InferenceData is assembled,
    Then the vector keeps its (1,) dim and the scalar has none.
    """
    # Arrange
    n_chains, n_draws = 3, 5
    rng = np.random.default_rng(0)
    stored_raw = {
        "vec": rng.normal(size=(n_chains, n_draws, 1)),
        "sca": rng.normal(size=(n_chains, n_draws)),
    }
    raw_start = {"vec": np.zeros(1), "sca": np.zeros(())}

    def identity(vec, sca):
        return [vec, sca]

    # Act
    idata = assemble_inference_data(
        stored_raw=stored_raw,
        stored_lp=np.zeros((n_chains, n_draws)),
        actual_draws=n_draws,
        n_chains=n_chains,
        raw_start=raw_start,
        raw_var_names=["vec", "sca"],
        out_var_names=["vec", "sca"],
        raw_to_phys_batched=identity,
        chain_seed_index=[0] * n_chains,
        label="test",
        log=logging.getLogger("test"),
    )

    # Assert
    assert idata.posterior["vec"].shape == (n_chains, n_draws, 1)
    assert idata.posterior["sca"].shape == (n_chains, n_draws)
    np.testing.assert_array_equal(
        idata.posterior["vec"].values, stored_raw["vec"]
    )
