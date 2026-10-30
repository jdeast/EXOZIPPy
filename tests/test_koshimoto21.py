"""The genulens port (galacticmodel/koshimoto21.py) against genulens itself.

tests/fixtures/genulens_points.json holds genulens's OWN functions evaluated
at fixed points (scripts/make_genulens_points.py, which compiles
scripts/genulens_reference_driver.cpp against a genulens build).  Pieces the
port reproduces analytically must agree to the fixture's 12 significant
digits; the tolerance of every other piece is stated where it is tested.
"""

import json
import os

import numpy as np
import pytest

from exozippy.components.galacticmodel import koshimoto21 as k21

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "genulens_points.json"
)
RTOL = 1e-11  # the fixture keeps 12 significant digits


@pytest.fixture(scope="module")
def points():
    with open(FIXTURE) as fh:
        return json.load(fh)


def _sight_lines(points):
    return [(name, s) for name, s in points["sight_lines"].items()]


def test_sgra_offset_matches_genulens(points):
    for _, s in _sight_lines(points):
        np.testing.assert_allclose(
            k21.SGRA_OFFSET, s["constants"]["xyzSgrA"], rtol=RTOL, atol=1e-12
        )


def test_positions_match_dlb2xyz(points):
    for name, s in _sight_lines(points):
        l_deg, b_deg = s["sightline"]["l"], s["sightline"]["b"]
        for row in s["density"]:
            xyz = k21.dlb_to_xyz(row["D"], l_deg, b_deg)
            np.testing.assert_allclose(
                xyz, row["xyz"], rtol=RTOL, atol=1e-9, err_msg=name
            )


def test_component_shapes_match_calc_rho_each(points):
    """Every analytic density shape (thin-disk age bins, thick disk, the E+E_X
    bar, the stellar halo) equals genulens's calc_rho_each at genulens's own
    positions."""
    n_checked = 0
    for name, s in _sight_lines(points):
        for row in s["density"]:
            x, y, z = row["xyz"]
            shapes = k21.log_component_shapes(x, y, z)
            for i, log_rho in enumerate(shapes):
                if log_rho is None:  # the NSD is tabulated
                    continue
                ref = row["rhos"][i]
                assert ref > 0.0, (name, row["D"], i)
                # compared in logs: far from a component (rhos ~ 1e-30) the
                # 12-digit rounding of the fixture's positions moves
                # log rho by ~1e-12 * |log rho|, i.e. rho by 4e-11 relative
                np.testing.assert_allclose(
                    log_rho,
                    np.log(ref),
                    rtol=RTOL,
                    atol=RTOL,
                    err_msg=f"{name} D={row['D']} {k21.COMPONENTS[i]}",
                )
                n_checked += 1
    assert n_checked > 500
