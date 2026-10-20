"""One DC2018 event in four architectures shares ONE prior (dc18_arch_config).

ln Z(2L1S) - ln Z(1L1S) is a statement about the data only when the two
models put the same prior on every parameter they share; a shared prior that
differs in support or normalization adds an arbitrary constant to the ratio.
examples/DC2018/dc18_arch_config.py builds the shared part with no
architecture argument and checks the WRITTEN files byte for byte.  These
tests check what that check cannot see: that the BUILT models agree,
parameter by parameter and element by element, on the bounds, prior centres,
widths, barrier scales and roles of everything they share -- including
neighbor_flux, whose bound the component used to take from the
seed-dependent flux bootstrap.

The event is synthetic (a PSPL in the full release's lc/ulwdc1_NNN layout,
magnitudes, inside the shipped Roman ephemeris), so the test needs no
challenge data on disk.
"""

import importlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.system import System

DC18 = Path(__file__).resolve().parents[1] / "examples" / "DC2018"

EVENT = 7
T0, TE, U0 = 2458385.0, 12.0, 0.2
M_SRC, M_BLEND = {"W149": 21.0, "Z087": 22.4}, {"W149": 21.6, "Z087": 23.0}


@pytest.fixture(scope="module")
def gen():
    sys.path.insert(0, str(DC18))
    try:
        yield importlib.import_module("dc18_arch_config")
    finally:
        sys.path.remove(str(DC18))


def _pspl_mag(t, band):
    u = np.hypot(U0, (t - T0) / TE)
    a = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    f = 10 ** (-0.4 * M_SRC[band]) * a + 10 ** (-0.4 * M_BLEND[band])
    return -2.5 * np.log10(f)


def _write_event(root):
    rng = np.random.default_rng(42)
    (root / "lc").mkdir(parents=True)
    (root / "event_info.txt").write_text(
        "ulwdc1_%03d %d 267.657 -27.937 8.29 2.75 0.33 5.29 0.63\n"
        % (EVENT, EVENT)
    )
    for band, step in (("W149", 0.4), ("Z087", 2.0)):
        t = np.arange(2458350.0, 2458420.0, step)
        err = np.full(t.size, 0.01)
        mag = _pspl_mag(t, band) + rng.normal(0.0, 0.01, t.size)
        np.savetxt(
            root / "lc" / ("ulwdc1_%03d_%s.txt" % (EVENT, band)),
            np.column_stack([t, mag, err]),
        )


@pytest.fixture(scope="module")
def written(gen, tmp_path_factory):
    """Given a synthetic DC2018 event, when the generator writes every
    architecture, provide {arch: config path}."""
    root = tmp_path_factory.mktemp("dc18data")
    _write_event(root)
    out = tmp_path_factory.mktemp("dc18arch")
    rc = gen.main(
        [
            "--event", str(EVENT), "--arch", "all", "--out", str(out),
            "--data-dir", str(root), "--cores", "1",
        ]
    )  # fmt: skip
    assert rc == 0
    return {
        arch: str(out / ("DC2018_%03d_%s.yaml" % (EVENT, arch)))
        for arch in gen.ARCHES
    }


def _build(cfg_path):
    with io.open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with io.open(cfg["parameter_file"], encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(cfg_path))
    try:
        system = System(cfg, user_params=params)
        system.prepare()
        system.model = system.build_model()
    finally:
        os.chdir(cwd)
    return system


@pytest.fixture(scope="module")
def built(written):
    return {arch: _build(path) for arch, path in written.items()}


# Elements every architecture shares, per component: the two stars (Lens,
# Source -- Source2 is element 2), the first source, both light curves, the
# event, the bands and the SED.  lens/planet are architecture-only.
SHARED_COMPONENTS = (
    "star",
    "source",
    "mulensinstrument",
    "mulensevent",
    "band",
)
# Architecture-specific by design (dc18_arch_config: the rho cap, the
# binary-source flux ratio and its log coordinate).
ARCH_ONLY = {
    "source.rho",
    "mulensinstrument.q_flux",
    "mulensinstrument.log_q_flux",
}
FIELDS = ("lower", "upper", "mu", "sigma", "bound_scale")


def _param_table(system):
    """{(component.param, element): (role, field values)} over the shared
    elements of the shared components."""
    n_shared = {"star": 2, "source": 1}
    out = {}
    for comp in SHARED_COMPONENTS:
        c = getattr(system, comp, None)
        if c is None:
            continue
        for attr in c.__dict__.values():
            if not hasattr(attr, "element_is_sampled") or not hasattr(
                attr, "build_pymc"
            ):
                continue
            name = attr.label
            if not name.startswith(comp + "."):
                continue
            if name in ARCH_ONLY:
                continue
            n = n_shared.get(comp, c.n_elements)
            for i in range(n):
                if not attr.element_is_active(i):
                    role = "inactive"
                elif attr.element_is_sampled(i):
                    role = "sampled"
                else:
                    role = "derived"
                vals = []
                for f in FIELDS:
                    v = getattr(attr, f, None)
                    if v is None:
                        vals.append(None)
                        continue
                    v = np.atleast_1d(np.asarray(v, dtype=float))
                    vals.append(float(v[i] if v.size > 1 else v[0]))
                pinned = vals[3] == 0.0
                if pinned:
                    v = np.atleast_1d(np.asarray(attr.initval, dtype=float))
                    vals.append(float(v[i] if v.size > 1 else v[0]))
                # "Never set" is spelled None (no vector) or NaN (a vector
                # some OTHER element set -- 1L2S's linked Source2 carries a
                # sigma vector); both mean the same, and NaN != NaN.
                vals = [
                    "unset" if v is None or np.isnan(v) else v for v in vals
                ]
                out[(name, i)] = (role, tuple(vals))
    return out


def test_written_files_share_one_prior(gen, written):
    """
    Given the four written architectures of one event,
    When check_shared reads them back,
    Then their shared views serialize to identical bytes, and each params
    file's header carries that digest.
    """
    digest = gen.check_shared(list(written.values()))
    for path in written.values():
        with io.open(path.replace(".yaml", ".params.yaml")) as fh:
            assert digest in fh.readline() + fh.readline()


def test_check_catches_a_changed_shared_prior(gen, written, tmp_path):
    """
    Given one architecture's params file with a shared bound moved,
    When check_shared compares it with an untouched one,
    Then it refuses, naming the files -- the check is not vacuous.
    """
    with io.open(written["2L1S"]) as fh:
        cfg = yaml.safe_load(fh)
    with io.open(cfg["parameter_file"]) as fh:
        params = yaml.safe_load(fh)
    params["mulensinstrument.Roman_W149.err_scale"]["upper"] = 3.0
    cfg["parameter_file"] = str(tmp_path / "bad.params.yaml")
    with io.open(cfg["parameter_file"], "w") as fh:
        yaml.safe_dump(params, fh)
    bad = tmp_path / "bad.yaml"
    with io.open(bad, "w") as fh:
        yaml.safe_dump(cfg, fh)
    with pytest.raises(SystemExit, match="shared priors differ"):
        gen.check_shared([written["1L1S"], str(bad)])


def test_seed_supplies_starts_never_priors(gen, written):
    """
    Given a seed whose entries carry bounds as well as initvals, and a 2L1S
    leaf a 1L1S model does not have,
    When seed_starts filters it for 1L1S,
    Then only initvals of parameters 1L1S has survive.
    """
    seed = {
        "source.Source.t_0": {"initval": T0, "lower": 0.0},
        "lens.Companion.log_s": {"initval": 0.2},
        "planet.Companion.log_q": {"initval": -3.0},
    }
    starts = gen.seed_starts("1L1S", seed, ev=None)
    assert starts == {"source.Source.t_0": {"initval": T0}}


@pytest.mark.slow  # four prepare + build_model passes (~70 s)
def test_built_models_share_every_shared_element(built):
    """
    Given the four architectures built (prepare + build_model),
    When every shared element's role, bounds, prior centre and width,
    barrier scale (and pin value) is tabulated,
    Then the tables are identical across architectures.
    """
    tables = {arch: _param_table(s) for arch, s in built.items()}
    ref = tables["1L1S"]
    # Non-vacuity: the table holds the parameters the prior set is about.
    names = {k[0] for k in ref}
    for must in (
        "source.t_0",
        "source.u0te",
        "mulensinstrument.err_scale",
        "mulensinstrument.log_f_total",
        "mulensinstrument.neighbor_flux",
        "star.av",
    ):
        assert must in names, must
    for arch, table in tables.items():
        common = set(ref) & set(table)
        assert common == set(ref), (arch, sorted(set(ref) - set(table)))
        diff = {k: (ref[k], table[k]) for k in common if ref[k] != table[k]}
        assert not diff, (arch, diff)


@pytest.mark.slow  # four prepare + build_model passes (~70 s)
def test_neighbor_flux_bound_is_the_data_median(built, gen):
    """
    Given the four built architectures (whose flux bootstraps differ: a
    second source and a companion change the decomposition),
    When neighbor_flux's upper bound is read,
    Then it is 2x the median observed flux of each light curve in all four.
    """
    for arch, system in built.items():
        inst = system.mulensinstrument
        med = np.asarray(inst.flux_median_init, dtype=float)
        assert np.allclose(
            np.asarray(inst.neighbor_flux.upper),
            2.0 * med,
            rtol=1e-12,
            atol=0.0,
        )


@pytest.mark.slow  # four prepare + build_model passes (~70 s)
def test_arch_specific_priors(built):
    """
    Given the built architectures,
    When the architecture-only parameters are read,
    Then 2L1S samples log_q on [-6, 0] and log_s on [-1, 1]; 1L2S samples
    log_q_flux on [-3, 3] per band; the rho cap is 1 on both, and 1L1S_fs
    keeps the defaults.yaml soft cap.
    """
    lens = built["2L1S"].lens
    planet = built["2L1S"].planet
    assert np.atleast_1d(planet.log_q.lower)[0] == -6.0
    assert np.atleast_1d(planet.log_q.upper)[0] == 0.0
    assert np.atleast_1d(lens.log_s.lower)[-1] == -1.0
    assert np.atleast_1d(lens.log_s.upper)[-1] == 1.0
    inst = built["1L2S"].mulensinstrument
    for i in range(inst.n_elements):
        assert inst.log_q_flux.element_is_sampled(i)
        assert not inst.q_flux.element_is_sampled(i)
    assert np.allclose(np.asarray(inst.log_q_flux.lower), -3.0)
    assert np.allclose(np.asarray(inst.log_q_flux.upper), 3.0)
    # The VALUE of the derived ratio, read off the built graph: q_flux is
    # 10**log_q_flux at the start (no constant of the test's own).
    model = built["1L2S"].model
    point = model.initial_point()
    (q_node, lq_node) = model.replace_rvs_by_values(
        [inst.q_flux.value, inst.log_q_flux.value]
    )
    f = pytensor.function(
        model.value_vars, [q_node, lq_node], on_unused_input="ignore"
    )
    q, lq = f(*[point[v.name] for v in model.value_vars])
    assert np.all(np.isfinite(lq))
    assert np.allclose(q, 10.0 ** np.asarray(lq), rtol=1e-12)
    for arch in ("2L1S", "1L2S"):
        assert np.allclose(np.asarray(built[arch].source.rho.upper), 1.0)
    assert np.allclose(np.asarray(built["1L1S_fs"].source.rho.upper), 10.0)
