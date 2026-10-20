"""The hogg-mixture policy: the optional Beta prior on out_frac and the
post-fit robust-likelihood audit (``<prefix>_outliers.txt``).

Background (components/instrument.md, "Robust observation likelihoods"):
on DC2018 no-planet plateau fits the mixture claimed the planetary-anomaly
points at ~11 nats each instead of ~600, subsidizing the wrong model.  The
Beta prior makes it pay per claimed point; the audit reports, per file, the
nats it forgave and how concentrated its outliers are in the event window.
"""

import types

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import integrate, stats

from conftest import _DummyConfigManager
from exozippy.components import likelihood as robust_support
from exozippy.components.instrument import Instrument
from exozippy.outputs import outliers


class _DummyInstrument(Instrument):
    @property
    def prefix(self):
        return "dummy"

    def register_parameters(self, system):  # pragma: no cover
        pass

    def build_likelihood(self, model, system):  # pragma: no cover
        pass


class _QuietConfigManager(_DummyConfigManager):
    def add_hint(self, path, value, rank=None):
        pass

    def add_scale_hint(self, path, scale):
        pass


class _FakeParam:
    def __init__(self, values):
        self.value = pt.as_tensor_variable(np.asarray(values, dtype=float))


def _make(config):
    return _DummyInstrument(config, config_manager=_QuietConfigManager())


# ---------------------------------------------------------------------------
# 1. The Beta prior: vocabulary and normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("uniform", None),
        ("flat", None),
        (False, None),
        ("beta", (1.0, 50.0)),
        (True, (1.0, 50.0)),
        ({"b": 20}, (1.0, 20.0)),
        ({"a": 2, "b": 3}, (2.0, 3.0)),
    ],
)
def test_out_frac_prior_spellings(value, expected):
    """``beta`` defaults to Beta(1, N/10); absent/uniform keeps flat."""
    assert robust_support.parse_out_frac_prior(value, 500) == expected


@pytest.mark.parametrize("bad", ["gamma", {"a": -1}, {"c": 1}, [1, 2]])
def test_out_frac_prior_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        robust_support.parse_out_frac_prior(bad, 100)


@pytest.mark.parametrize(
    "a, b, lower, upper",
    [
        (1.0, 50.0, 0.0, 0.5),  # the default on a 500-point file
        (1.0, 0.4, 0.0, 0.5),  # b < 1: density rises toward the cap
        (0.5, 5.0, 0.0, 0.5),  # a < 1: integrable 1/sqrt(f) spike
        (2.0, 3.0, 0.1, 0.4),  # a user-tightened support
        (3.0, 2.0, 0.6, 0.9),  # both bounds above the median (betaincc path)
    ],
)
def test_beta_prior_is_a_normalized_density_on_the_support(a, b, lower, upper):
    """
    Given Beta shapes and out_frac's [lower, upper],
    When the potential's density is integrated over that interval,
    Then it integrates to 1 (a PROPER prior), and pointwise equals scipy's
    Beta logpdf minus the log of the mass on the interval.
    """
    f = pt.dscalar("f")
    fn = pytensor.function(
        [f], robust_support.beta_logp(f, a, b, lower=lower, upper=upper)
    )
    total, _ = integrate.quad(lambda x: np.exp(fn(x)), lower, upper, limit=200)
    assert total == pytest.approx(1.0, abs=1e-7)

    mass = stats.beta.cdf(upper, a, b) - stats.beta.cdf(lower, a, b)
    for x in np.linspace(lower, upper, 7)[1:-1]:
        assert fn(x) == pytest.approx(
            stats.beta.logpdf(x, a, b) - np.log(mass), rel=1e-10
        )


def test_out_frac_prior_on_a_non_hogg_file_raises():
    with pytest.raises(ValueError, match="likelihood: hogg"):
        _make([{"file": "a.dat", "out_frac_prior": "beta"}])


def test_no_out_frac_prior_key_adds_no_prior():
    """The default stays the flat prior: no shapes, no extra potential."""
    inst, model, _ = _dispatched(prior=None)
    assert inst._out_frac_beta == {}
    assert [v.name for v in model.potentials] == ["m.hogg.1"]


def _dispatched(prior, n_per=50):
    rng = np.random.default_rng(3)
    inst_map = np.repeat([0, 1], n_per)
    sigma = np.full(2 * n_per, 0.2)
    mu = np.zeros(2 * n_per)
    y = rng.normal(0, sigma)
    cfg_b = {"file": "b.dat", "likelihood": "hogg"}
    if prior is not None:
        cfg_b["out_frac_prior"] = prior
    inst = _make([{"file": "a.dat"}, cfg_b])
    inst.inst_map = inst_map
    inst.n_total_obs = 2 * n_per
    inst._prepare_robust(sigma, inst_map)
    with pm.Model() as model:
        inst.out_frac = _FakeParam([0.0, 0.03])
        inst.out_scale = _FakeParam([0.0, 2.0])
        inst.add_observation_likelihood(
            "m",
            mu=pt.as_tensor_variable(mu),
            sigma=pt.as_tensor_variable(sigma),
            observed=y,
        )
    return inst, model, (y, mu, sigma)


def test_beta_prior_adds_exactly_its_normalized_term():
    """
    Given the second file with ``out_frac_prior: beta`` (50 points),
    When the likelihood is built,
    Then the shapes are Beta(1, 5), one named prior Potential is added, and
    the model logp differs from the flat-prior model by exactly the
    normalized Beta logpdf at out_frac.
    """
    inst, model, _ = _dispatched(prior="beta")
    assert inst._out_frac_beta == {1: (1.0, 5.0)}
    assert "dummy.out_frac.beta.1" in {v.name for v in model.potentials}

    _, flat, _ = _dispatched(prior=None)
    diff = float(model.compile_logp()({})) - float(flat.compile_logp()({}))
    expected = stats.beta.logpdf(0.03, 1.0, 5.0) - np.log(
        stats.beta.cdf(0.5, 1.0, 5.0)
    )
    assert diff == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. The audit on a synthetic light curve
# ---------------------------------------------------------------------------
T0, TE = 50.0, 5.0


class _Event:
    """Stands in for the component that owns the signal's timescale."""

    prefix = "event"

    def event_windows(self, system, point):
        return [(T0 - TE, T0 + TE)]


def _light_curve_system(kind, n=500, n_bad=8, out_frac=0.05):
    """A flat light curve, model = 1 exactly, with ``n_bad`` 30-sigma
    excursions either CLUSTERED at the event (a missed anomaly) or
    SCATTERED evenly over the season (bad data / variability)."""
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 100.0, n)
    err = np.full(n, 0.01)
    y = 1.0 + rng.normal(0.0, 0.01, n)
    if kind == "clustered":
        bad = np.flatnonzero(np.abs(t - T0) < 2.0)[:n_bad]
    elif kind == "scattered":
        bad = np.arange(n_bad) * (n // n_bad) + 3
    else:
        bad = np.array([], dtype=int)
    y[bad] += 0.3

    inst = _make([{"name": "W149", "file": "w.dat", "likelihood": "hogg"}])
    inst.inst_map = np.zeros(n, dtype=int)
    inst.n_total_obs = n
    inst.time = t
    inst._prepare_robust(err, inst.inst_map)
    with pm.Model():
        inst.out_frac = _FakeParam([out_frac])
        inst.out_scale = _FakeParam([10 * 0.01])
        inst.add_observation_likelihood(
            "m",
            mu=pt.as_tensor_variable(np.ones(n)),
            sigma=pt.as_tensor_variable(err),
            observed=y,
        )
    system = types.SimpleNamespace(
        active_components={"dummy": inst, "event": _Event()},
        plot_params=[],
        plot_branch_labels=set(),
    )
    return system, (t, y, err, bad)


def test_audit_flags_a_clustered_anomaly_by_its_enrichment():
    """
    Given eight 30-sigma points inside |t - t0| < 1 tE (10% of the season),
    When the audit runs,
    Then all eight are outlier-component members, enrich = 1/0.1 = 10, and
    the file is flagged for enrichment AND for the nats forgiven.
    """
    system, _ = _light_curve_system("clustered")
    (row,) = outliers.audit_point(system, {}, "MAP")
    assert row["n_high"] == 8
    t = np.linspace(0.0, 100.0, 500)
    base = np.mean(np.abs(t - T0) <= TE)
    assert row["enrich"] == pytest.approx(1.0 / base, rel=1e-12)
    assert row["delta_logl"] < -robust_support.FORGIVEN_NATS_FLAG
    assert any("concentrated" in f for f in row["flags"])
    assert any("forgives" in f for f in row["flags"])


def test_audit_does_not_call_scattered_outliers_an_anomaly():
    """
    Given the same eight excursions spread evenly over the season,
    When the audit runs,
    Then enrich ~ 1 (one of the eight happens to fall in the window: 1.25)
    and the only flag is the forgiven-nats one -- the case DC2026's
    starspot fits need the mixture for.
    """
    system, (t, _, _, bad) = _light_curve_system("scattered")
    (row,) = outliers.audit_point(system, {}, "MAP")
    assert row["n_high"] == 8
    inside = np.abs(t - T0) <= TE
    expected = np.mean(inside[bad]) / np.mean(inside)
    assert row["enrich"] == pytest.approx(expected, rel=1e-12)
    assert row["enrich"] < robust_support.ENRICH_FLAG
    assert not any("concentrated" in f for f in row["flags"])
    assert any("forgives" in f for f in row["flags"])


def test_audit_reports_the_mixture_numbers_in_closed_form():
    """Gaussian and hogg logL are the file's closed forms, same sigma."""
    system, (t, y, err, _) = _light_curve_system("scattered")
    (row,) = outliers.audit_point(system, {}, "median")
    r = y - 1.0
    gauss = np.sum(stats.norm.logpdf(r, 0.0, err))
    wide = np.sqrt(err**2 + 0.1**2)
    hogg = np.sum(
        np.logaddexp(
            np.log(0.95) + stats.norm.logpdf(r, 0.0, err),
            np.log(0.05) + stats.norm.logpdf(r, 0.0, wide),
        )
    )
    assert row["gauss_logl"] == pytest.approx(gauss, rel=1e-10)
    assert row["hogg_logl"] == pytest.approx(hogg, rel=1e-10)
    assert row["delta_logl"] == pytest.approx(gauss - hogg, rel=1e-9)


def test_audit_is_quiet_on_clean_data():
    """No outliers: nothing claimed, enrich nan, no flag."""
    system, _ = _light_curve_system("clean")
    (row,) = outliers.audit_point(system, {}, "MAP")
    assert row["n_high"] == 0
    assert np.isnan(row["enrich"])
    assert row["flags"] == []


def test_write_robust_audit_writes_the_table_and_warns(
    tmp_path, monkeypatch, caplog
):
    system, _ = _light_curve_system("clustered")
    monkeypatch.setattr(outliers, "audit_points", lambda s, i: {"MAP": {}})
    with caplog.at_level("WARNING"):
        rows = outliers.write_robust_audit(system, None, tmp_path / "fit")
    text = (tmp_path / "fit_outliers.txt").read_text()
    assert len(rows) == 1
    assert "dummy.W149" in text and "FLAG:" in text
    assert "ROBUST LIKELIHOOD" in caplog.text


def test_write_robust_audit_is_silent_without_the_mixture(tmp_path):
    inst = _make([{"file": "a.dat"}])
    system = types.SimpleNamespace(active_components={"dummy": inst})
    assert outliers.write_robust_audit(system, None, tmp_path / "fit") is None
    assert not (tmp_path / "fit_outliers.txt").exists()


def test_enrichment_ratio_edge_cases():
    t = np.linspace(0, 10, 11)
    m = np.zeros(11)
    assert np.isnan(robust_support.enrichment_ratio(m, t, [])[0])
    m[5] = 0.9
    enrich, n = robust_support.enrichment_ratio(m, t, [(4.5, 5.5)])
    assert n == 1 and enrich == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# 3. The microlensing event's window hook
# ---------------------------------------------------------------------------
def test_mulensevent_window_is_one_tE_around_each_source_t0_with_a_floor():
    from exozippy.components.mulensing.mulensevent import MulensEvent

    def param(label, initval):
        return types.SimpleNamespace(label=label, initval=np.array(initval))

    event = types.SimpleNamespace(
        t_E=param("mulensevent.t_E", [30.0]),
        EVENT_WINDOW_MIN_HALF_WIDTH=MulensEvent.EVENT_WINDOW_MIN_HALF_WIDTH,
    )
    system = types.SimpleNamespace(
        source=types.SimpleNamespace(
            t_0=param("source.t_0", [2460000.0, 2460010.0])
        )
    )
    point = {"mulensevent.t_E": np.array([12.0])}
    assert MulensEvent.event_windows(event, system, point) == [
        (2459988.0, 2460012.0),
        (2459998.0, 2460022.0),
    ]
    short = {"mulensevent.t_E": np.array([0.2])}
    lo, hi = MulensEvent.event_windows(event, system, short)[0]
    assert hi - lo == pytest.approx(
        2 * MulensEvent.EVENT_WINDOW_MIN_HALF_WIDTH
    )


# ---------------------------------------------------------------------------
# 4. A real System: the prior reaches the model and the Prior column
# ---------------------------------------------------------------------------
def _write_rv(path, seed, n=40):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, n))
    rv = 30.0 * np.sin(2 * np.pi * t / 17.0) + rng.normal(0, 3.0, n)
    np.savetxt(path, np.column_stack([t, rv, np.full(n, 3.0)]))
    return str(path)


def _rv_system(files, prior):
    from exozippy.system import System

    inst = [
        {"name": "A_inst", "file": files[0], "likelihood": "hogg"},
        {"name": "B_inst", "file": files[1]},
    ]
    if prior is not None:
        inst[0]["out_frac_prior"] = prior
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": inst,
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(17.0)},
        "orbit.b.tc": {"initval": 2455010.0},
        "orbit.b.secosw": {"initval": 0.0},
        "orbit.b.sesinw": {"initval": 0.0},
    }
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


@pytest.mark.slow
def test_rv_system_beta_prior_is_the_only_change(tmp_path):
    """
    Given the same two-file RV system with and without ``out_frac_prior:
    beta`` on its hogg file (40 points -> Beta(1, 4)),
    When both models are built,
    Then the logp at the shared start differs by exactly the normalized
    Beta logpdf at the start out_frac, the Prior column of that element
    says Beta, and the audit hook reports the file with its epochs.
    """
    files = [
        _write_rv(tmp_path / "a.rv", 21),
        _write_rv(tmp_path / "b.rv", 22),
    ]
    sys_beta, m_beta = _rv_system(files, "beta")
    sys_flat, m_flat = _rv_system(files, None)
    rv = sys_beta.rvinstrument
    assert rv._out_frac_beta == {0: (1.0, 4.0)}

    point = m_flat.initial_point()
    diff = float(m_beta.compile_logp()(point)) - float(
        m_flat.compile_logp()(point)
    )
    f0 = float(np.atleast_1d(rv.out_frac.initval)[0])
    expected = stats.beta.logpdf(f0, 1.0, 4.0) - np.log(
        stats.beta.cdf(0.5, 1.0, 4.0)
    )
    assert diff == pytest.approx(expected, rel=1e-9)

    texts = [c.text for c in rv.out_frac.prior_contributions_at(0)]
    assert texts == ["Beta(1, 4)"]
    assert rv.out_frac.prior_contributions_at(1) == []

    (summary,) = rv.robust_fit_summary(sys_beta, point)
    assert summary["name"] == "A_inst" and summary["n"] == 40
    assert summary["times"].shape == (40,)
    assert np.isfinite(summary["gauss_logl"]) and np.isfinite(
        summary["hogg_logl"]
    )
