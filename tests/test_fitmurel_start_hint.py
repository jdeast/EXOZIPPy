"""The `fitmurel: true` start of mu_rel = 0 is a ranked add_hint (2.1.26(d)).

It used to be an "overrides"-channel initval on the manifest entry.  That
reached resolve() but never the provenance ledger, so the startup table,
export_solution and the GUI could not say where the start came from.  It is
now ``add_hint(..., rank=PRECEDENCE_DEFAULT)``: a modelling default, beaten by
a user initval and by any engine-solved value.  With no user value the start
logp is bit-identical to the overrides version on every shipped example that
samples mu_rel (the DC2018 sweep4/sweep5 configs and events/152/obs).
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.config import PRECEDENCE_DEFAULT, PRECEDENCE_USER
from exozippy.system import System

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"
_WORKDIR = None


def _kmt_workdir():
    """A private copy of the example dir for THIS module (xdist isolation)."""
    global _WORKDIR
    if _WORKDIR is None:
        _WORKDIR = Path(tempfile.mkdtemp(prefix="kmt_murel_")) / "KMT"
        shutil.copytree(_KMT_DIR, _WORKDIR)
    return _WORKDIR


def _prepare(extra_params=None):
    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")
    cwd = os.getcwd()
    os.chdir(_kmt_workdir())
    try:
        with open("KMT-2019-BLG-1806.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)
        config["mulensevent"][0]["fitmurel"] = True
        user_params.update(extra_params or {})
        system = System(config, user_params=user_params)
        system.prepare()
    finally:
        os.chdir(cwd)
    return system


@pytest.fixture(scope="module")
def no_user():
    return _prepare()


@pytest.fixture(scope="module")
def with_user():
    return _prepare({"mulensevent.0.mu_ra_rel": {"initval": 3.25}})


def test_start_is_a_ranked_hint_not_an_override(no_user):
    """
    Given fitmurel and no user mu_rel entry,
    When the system is prepared,
    Then both components carry a PRECEDENCE_DEFAULT hint of 0, the manifest
    entry has no "overrides" initval, and nothing reached user_params.
    """
    cm = no_user.config_manager
    ev = no_user.mulensevent
    for p in ("mu_ra_rel", "mu_dec_rel"):
        path = f"mulensevent.0.{p}"
        assert cm.hint_ranks[path] == PRECEDENCE_DEFAULT
        assert float(cm.hints[path]) == 0.0
        entry = ev.manifest[p]
        assert not (entry or {}).get("overrides"), entry
        # The engine's own inject-back of a solved start is flagged
        # derived; an unflagged entry would be a component write.
        for k, v in cm.user_params.items():
            if p in k:
                assert isinstance(v, dict) and v.get("derived"), (k, v)


def test_ledger_reports_the_hint_not_the_user(no_user):
    """
    Given no user mu_rel entry,
    Then the provenance ledger files the start at a non-user rank (the
    hint's, or an engine-solved one above it) and initval_source never says
    "user".
    """
    cm = no_user.config_manager
    for p in ("mu_ra_rel", "mu_dec_rel"):
        rank = cm._last_provenance[f"mulensevent.0.{p}"]
        assert PRECEDENCE_DEFAULT <= rank < PRECEDENCE_USER
        assert cm.initval_source("mulensevent", p, element=0) != "user"


def test_user_initval_beats_the_hint(with_user):
    """
    Given a params-file initval on mu_ra_rel,
    Then the resolved start is the user's and the ledger says so, while the
    untouched mu_dec_rel still starts from the hint.
    """
    cm = with_user.config_manager
    assert cm._last_provenance["mulensevent.0.mu_ra_rel"] == PRECEDENCE_USER
    assert cm.initval_source("mulensevent", "mu_ra_rel", element=0) == "user"
    # mas/yr is both the user and the internal unit (defaults.yaml).
    cfg = cm.resolve("mulensevent", "mu_ra_rel")
    assert float(np.ravel(cfg["initval"])[0]) == pytest.approx(3.25)
    assert cm.initval_source("mulensevent", "mu_dec_rel", element=0) != "user"
