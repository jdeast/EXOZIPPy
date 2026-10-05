"""Small microlensing robustness fixes (review 2.6.7).

Each test builds the synthetic system in tests/mulens_synthetic.py with the
topology its item needs.
"""

import pytest
from mulens_synthetic import mulens_config, mulens_params, write_flat_lc

from exozippy.system import System

# ---------------------------------------------------------------------------
# 2.6.7: microlensing photometry without its event raises a CONFIG error
# naming the missing block (it used to die with a bare AttributeError on
# system.source deep inside stage 1).
# ---------------------------------------------------------------------------


def test_mulens_photometry_without_an_event_names_the_missing_blocks(
    tmp_path,
):
    """
    Given: a config with a mulensinstrument light curve and stars, but no
      mulensevent / lens / source block,
    When: prepare() runs stage 1,
    Then: MulensInstrument.load_data raises a ValueError naming all three
      missing blocks -- not an AttributeError on `system.source`.
    """
    # Arrange
    full = mulens_config(write_flat_lc(tmp_path / "lc.dat"))
    params = mulens_params(full)
    config = {k: full[k] for k in ("star", "mulensinstrument")}
    system = System(config, user_params=params)

    # Act / Assert
    with pytest.raises(ValueError) as err:
        system.prepare()
    msg = str(err.value)
    for block in ("'mulensevent'", "'lens'", "'source'"):
        assert block in msg, msg
