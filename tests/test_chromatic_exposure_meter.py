"""Per-order rows and the chromatic exposure meter (rvinstrument
``data_format: raw_redshift``; ``src/exozippy/components/rvinstrument/
rvinstrument.md``, "Per-order rows"; ``docs/raw_redshift_data.md``).

Pinned here:

* a simulated CHROMATIC exposure -- three orders whose flux ramps up, stays
  flat and ramps down across an hour, white light uniform -- is recovered to
  < 1 mm/s when each order is averaged with its own meter channel, and the
  white-light (achromatic) record leaves exactly the residual the forward
  model predicts for those wrong weights (meters per second here);
* the fallback ladder per row: own channel -> the exposure's white-light
  bins -> the exptime_s column -> the tag, each a stated default;
* the spelling's refusals (a channel bin without a channel column, a
  channel no row names, several rows of one exposure without a channel
  column, an err_shared that differs inside one exposure or meets gp:);
* the shared-noise likelihood against a dense multivariate normal, alone
  and inside a built model.

The forward model is ``examples/raw_redshift_sim/make_data.py``'s, which the
raw-redshift likelihood already reproduces to < 1 mm/s
(``tests/test_raw_redshift.py``); here it is evaluated on each row's node
table, the part this feature changes.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from exozippy.components.exposure import weighted_exposure_rule
from exozippy.components.instrument import Instrument

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "raw_redshift_sim"


def _load_make_data():
    spec = importlib.util.spec_from_file_location(
        "raw_redshift_make_data_chromatic", EXAMPLE / "make_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MD = _load_make_data()
SITE = "HPF"
STEP = MD.STEP_S
#: Three orders: their channel labels (nm) and the fractional flux slope
#: across the exposure.  The slopes sum to zero, so the white light is
#: uniform and its photon-weighted midpoint is the tag.
CHANNELS = {450.0: 0.8, 550.0: 0.0, 700.0: -0.8}
N_STEPS = 120  # 60 min
N_EPOCHS = 5


def _exposure(t_mid, slope):
    """One ``MD.exposures``-style entry: 30-s step starts (JD_UTC) and the
    flux, ``1 + slope x`` with x from -1/2 to 1/2 across the exposure,
    placed so the WHITE light (uniform) has its mean at ``t_mid``."""
    k = np.arange(N_STEPS)
    x = (k + 0.5) / N_STEPS - 0.5
    flux = 1.0 + slope * x
    mean = 0.5 * N_STEPS * STEP
    return t_mid + (k * STEP - mean) / 86400.0, flux


def _meter_lines(t_mid, exp, channel=None):
    starts, flux = exp
    out = []
    for t0, f in zip(starts, flux):
        line = f"{t_mid:.8f} {t0 + 0.5 * STEP / 86400.0:.8f} {STEP:.1f} {f * 1e4:.4f}"
        out.append(line if channel is None else f"{line} {channel:.1f}")
    return out


def _entry(data, meter=None, **columns):
    cols = {"time": 0, "rv": 1, "err": 2, "exptime_s": 3, "channel": 4}
    cols.update(columns)
    e = {
        "name": SITE,
        "file": str(data),
        "data_format": "raw_redshift",
        "observer_location": MD.SITES[SITE],
        "time_frame": "jd",
        "time_scale": "utc",
        "epoch": MD.TRUTH["epoch"],
        "columns": {k: v for k, v in cols.items() if v is not None},
    }
    if meter is not None:
        e["exposure_meter"] = str(meter)
    return e


def _prepare(entry, extra_params=None):
    from exozippy.system import System

    p = {
        "star.A.mass": {"initval": 0.64, "sigma": 0.03},
        "star.A.radius": {"initval": 0.62, "sigma": 0.03},
        "star.A.ra": {"initval": MD.TRUTH["ra"]},
        "star.A.dec": {"initval": MD.TRUTH["dec"]},
        "star.A.rv": {"mu": MD.TRUTH["rv"], "sigma": 500.0},
        "orbit.b.period": {"initval": MD.TRUTH["period"]},
        "orbit.b.tc": {"initval": MD.TRUTH["tc"]},
    }
    p.update(extra_params or {})
    cfg = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [entry],
    }
    system = System(cfg, user_params=p)
    system.prepare()
    return system


def _node_model(system):
    """The raw file's prediction from its NODE TABLE: make_data's numpy
    forward model at every node (observer TDB, the observer terms computed
    at load), averaged with the node weights -- exactly what the graph's
    node_mean does to its own instantaneous model."""
    from exozippy.components.rvinstrument.redshift import predicted_cz

    rec = system.rvinstrument._raw_files[0]
    star = (
        np.radians(MD.TRUTH["ra"]),
        np.radians(MD.TRUTH["dec"]),
        MD.TRUTH["pm_ra"],
        MD.TRUTH["pm_dec"],
        MD.TRUTH["parallax"],
        MD.TRUTH["rv"],
    )
    cz, _ = predicted_cz(
        rec["node_t_tdb"],
        rec["node_obs"],
        star,
        MD.TRUTH["epoch"],
        z0=MD.ZERO_POINTS[SITE] / 299792458.0,
        v_reflex=lambda tb: (
            MD.keplerian_rv(MD.emission_time(tb)),
            None,
            None,
        ),
        phi_star=lambda tb: MD.companion_potential(MD.emission_time(tb)),
        v_perp2=lambda tb: MD.transverse_speed2(MD.emission_time(tb)),
    )
    return np.bincount(
        rec["node_row"],
        weights=rec["node_w"] * cz,
        minlength=rec["t_tdb"].size,
    )


@pytest.fixture(scope="module")
def chromatic(tmp_path_factory):
    """Per-order rows of N_EPOCHS hour-long HPF exposures, each order's
    noise-free c*z the photon mean over ITS OWN flux; the meter written
    both per channel and as white light (the channels' sum)."""
    tmp = tmp_path_factory.mktemp("chromatic")
    tags = np.loadtxt(EXAMPLE / f"Gaia4like.{SITE}.rv")[:N_EPOCHS, 0]
    truth, rows, per_chan, achrom = {}, [], [], []
    for lam, slope in CHANNELS.items():
        exps = [_exposure(t, slope) for t in tags]
        truth[lam] = MD.predict(SITE, tags, exps=exps)[0]
    white_exps = [_exposure(t, 0.0) for t in tags]
    white_cz = MD.predict(SITE, tags, exps=white_exps)[0]
    for k, t in enumerate(tags):
        white_flux = np.zeros(N_STEPS)
        for lam, slope in CHANNELS.items():
            exp = _exposure(t, slope)
            rows.append((t, truth[lam][k], 1.0, N_STEPS * STEP, lam, 0.5))
            per_chan += _meter_lines(t, exp, lam)
            white_flux = white_flux + exp[1]
        achrom += _meter_lines(t, (exp[0], white_flux))
    fmt = "%.8f %.6f %.2f %.1f %.1f %.2f"
    data = tmp / "orders.rv"
    np.savetxt(data, np.array(rows), fmt=fmt)
    m_chan = tmp / "chan.expmeter"
    m_chan.write_text("\n".join(per_chan) + "\n")
    m_white = tmp / "white.expmeter"
    m_white.write_text("\n".join(achrom) + "\n")
    return {
        "tmp": tmp,
        "tags": tags,
        "data": data,
        "rows": np.array(rows),
        "meter_chan": m_chan,
        "meter_white": m_white,
        "truth": truth,
        "white": white_cz,
    }


def _expected(rows, by_channel):
    """Row-ordered expectation from a {channel: (N_EPOCHS,)} table (rows
    are written epoch-major, channel-minor, already time sorted)."""
    lams = list(CHANNELS)
    return np.array(
        [
            by_channel[r[4]][k // len(lams)]
            if isinstance(by_channel, dict)
            else by_channel[k // len(lams)]
            for k, r in enumerate(rows)
        ]
    )


def test_per_channel_averaging_recovers_a_chromatic_exposure(chromatic):
    """
    Given per-order rows of hour-long exposures whose blue order's flux
      ramps up, red order's ramps down and green is flat (white light
      uniform), with the meter recorded per channel,
    When the raw file is prepared and its node table evaluated with the
      forward model that made the data,
    Then every row reproduces its own order's photon-weighted c*z to
      < 1 mm/s -- and the orders genuinely differ: blue and red are meters
      per second apart, so the test is not vacuous.
    """
    s = _prepare(_entry(chromatic["data"], chromatic["meter_chan"]))
    model = _node_model(s)
    rows = chromatic["rows"]
    truth = _expected(rows, chromatic["truth"])
    assert np.max(np.abs(model - truth)) < 1e-3
    spread = np.abs(chromatic["truth"][450.0] - chromatic["truth"][700.0])
    assert np.max(spread) > 1.0


def test_white_light_averaging_leaves_the_chromatic_residual(chromatic):
    """
    Given the same rows with the meter written as white light only (four
      fields, the channels' summed counts),
    When prepared and evaluated the same way,
    Then every order is averaged with the white-light weights: the model
      equals the white-light forward model to < 1 mm/s, so each order is
      off its truth by exactly the predicted chromatic residual -- up to
      meters per second for the ramped orders, ~0 for the flat one.
    """
    s = _prepare(_entry(chromatic["data"], chromatic["meter_white"]))
    model = _node_model(s)
    rows = chromatic["rows"]
    truth = _expected(rows, chromatic["truth"])
    white = _expected(rows, chromatic["white"])
    assert np.max(np.abs(model - white)) < 1e-3
    resid = model - truth
    blue = rows[:, 4] == 450.0
    green = rows[:, 4] == 550.0
    assert np.max(np.abs(resid[blue])) > 1.0
    assert np.max(np.abs(resid[green])) < 1e-3


def _rule(meter_lines, t_conv):
    """The node table one row of a meter group should get: the bins'
    3-point box sub-samples, the 3-node Gauss rule of them."""
    from exozippy.components.exposure import gauss_legendre_rule

    x3, w3 = gauss_legendre_rule(3)
    tab = np.array([[float(v) for v in ln.split()[:4]] for ln in meter_lines])
    sub_t = (
        t_conv(tab[:, 1])[:, None] + x3[None, :] * tab[:, 2][:, None] / 86400.0
    )
    sub_c = tab[:, 3][:, None] * w3[None, :]
    return weighted_exposure_rule(sub_t.ravel(), sub_c.ravel(), 3)


def test_the_fallback_ladder_per_row(chromatic):
    """
    Given four exposures of three orders, every row with an exptime_s, and
      a meter that has
      exposure 0: channels 450 and 550 AND a white-light record;
      exposure 1: the white-light record only;
      exposure 2: channel 450 only;
      exposure 3: nothing,
    When the node table is built,
    Then each row takes the most complete description it has, in order:
      its own channel (exposure 0, 450/550; exposure 2, 450) -> the
      exposure's white-light bins (exposure 0, 700; exposure 1, all) ->
      the exptime_s column, Gauss-Legendre (exposure 2, 550/700; exposure
      3, all).  The last rung, one node at the tag, is the next test's.
    """
    from exozippy.components.exposure import gauss_legendre_rule

    tmp = chromatic["tmp"]
    tags = chromatic["tags"][:4]
    lines, groups = [], {}
    with open(tmp / "ladder.rv", "w") as fh:
        for k, t in enumerate(tags):
            white_flux = np.zeros(N_STEPS)
            for lam, slope in CHANNELS.items():
                exp = _exposure(t, slope)
                white_flux = white_flux + exp[1]
                fh.write(f"{t:.8f} 0.0 1.0 {N_STEPS * STEP:.1f} {lam:.1f}\n")
                if (k == 0 and lam != 700.0) or (k == 2 and lam == 450.0):
                    groups[(k, lam)] = _meter_lines(t, exp, lam)
                    lines += groups[(k, lam)]
            if k in (0, 1):
                groups[(k, None)] = _meter_lines(t, (exp[0], white_flux))
                lines += groups[(k, None)]
    meter = tmp / "ladder.expmeter"
    meter.write_text("\n".join(lines) + "\n")
    s = _prepare(_entry(tmp / "ladder.rv", meter))
    rvi = s.rvinstrument
    rec = rvi._raw_files[0]

    def t_conv(t):
        return rvi._to_observer_tdb(t, 0, rvi.observers[0])

    x_u, w_u = gauss_legendre_rule(3)
    lams = list(CHANNELS)
    rungs = []
    for j in range(rec["t_tdb"].size):
        k, c = divmod(j, len(lams))
        lam = lams[c]
        sel = rec["node_row"] == j
        if (k, lam) in groups:
            want, rung = _rule(groups[(k, lam)], t_conv), "channel"
        elif (k, None) in groups:
            want, rung = _rule(groups[(k, None)], t_conv), "white"
        else:
            want = (rec["t_tdb"][j] + x_u * N_STEPS * STEP / 86400.0, w_u)
            rung = "exptime_s"
        rungs.append(rung)
        np.testing.assert_allclose(
            rec["node_t_tdb"][sel], want[0], rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(rec["node_w"][sel], want[1], atol=1e-12)
    # Every rung fired, on the rows the docstring names.
    assert rungs == [
        "channel", "channel", "white",
        "white", "white", "white",
        "channel", "exptime_s", "exptime_s",
        "exptime_s", "exptime_s", "exptime_s",
    ]  # fmt: skip


def test_a_row_with_no_description_is_the_tag_and_warns(chromatic, caplog):
    """
    Given per-order rows with no exposure columns and a meter covering only
      the first exposure (white light),
    When prepared,
    Then the other exposures' rows are one node at the tag (the last rung
      of the ladder) and the warning names them.
    """
    tmp = chromatic["tmp"]
    tags = chromatic["tags"][:2]
    with open(tmp / "bare.rv", "w") as fh:
        for t in tags:
            for lam in CHANNELS:
                fh.write(f"{t:.8f} 0.0 1.0 {lam:.1f}\n")
    meter = tmp / "bare.expmeter"
    meter.write_text(
        "\n".join(_meter_lines(tags[0], _exposure(tags[0], 0.0))) + "\n"
    )
    with caplog.at_level("WARNING"):
        s = _prepare(
            _entry(
                tmp / "bare.rv",
                meter,
                exptime_s=None,
                channel=3,
                err_shared=None,
            )
        )
    rec = s.rvinstrument._raw_files[0]
    counts = np.bincount(rec["node_row"])
    assert np.all(counts[:3] == 3) and np.all(counts[3:] == 1)
    assert "no bins for 3 data row(s)" in caplog.text


@pytest.mark.parametrize(
    "case, match",
    [
        ("channel_bins_without_column", "names no 'channel' column"),
        ("unknown_channel", r"channel\(s\) \[999.0\] name no row"),
        ("two_rows_without_channel", "need a 'channel' column"),
        ("six_fields", "6 fields"),
    ],
)
def test_the_meter_spelling_refusals(chromatic, case, match):
    """
    Given a meter/data pair with one spelling error,
    When prepared,
    Then it raises naming it: a per-channel bin needs the data file's
      channel column to be assigned; a channel no row names is a typo
      that would drop its record; several rows sharing one tag are orders
      of one exposure only with a channel column; and a line is four or
      five fields.
    """
    tmp = chromatic["tmp"]
    t = chromatic["tags"][0]
    exp = _exposure(t, 0.0)
    data = tmp / f"{case}.rv"
    meter = tmp / f"{case}.expmeter"
    columns = {"err_shared": None}
    if case == "channel_bins_without_column":
        data.write_text(f"{t:.8f} 0.0 1.0 3600.0\n")
        meter.write_text("\n".join(_meter_lines(t, exp, 450.0)) + "\n")
        columns["channel"] = None
    elif case == "unknown_channel":
        data.write_text(f"{t:.8f} 0.0 1.0 3600.0 450.0\n")
        meter.write_text("\n".join(_meter_lines(t, exp, 999.0)) + "\n")
    elif case == "two_rows_without_channel":
        data.write_text(f"{t:.8f} 0.0 1.0 3600.0\n{t:.8f} 1.0 1.0 3600.0\n")
        meter.write_text("\n".join(_meter_lines(t, exp)) + "\n")
        columns["channel"] = None
    else:
        data.write_text(f"{t:.8f} 0.0 1.0 3600.0 450.0\n")
        meter.write_text(_meter_lines(t, exp, 450.0)[0] + " 7\n")
    with pytest.raises(ValueError, match=match):
        _prepare(_entry(data, meter, **columns))


# ---------------------------------------------------------------------------
# The correlated-error (shared-noise) likelihood
# ---------------------------------------------------------------------------


def test_shared_noise_logp_is_the_dense_multivariate_normal():
    """
    Given residuals in exposures of 1, 3 and 5 rows with their own
      diagonal errors and a per-exposure shared variance (one of them 0),
    When ``Instrument.shared_noise_logp`` is evaluated,
    Then it equals scipy's multivariate normal with the dense block
      covariance diag(sigma^2) + s2_g 1 1^T to 1e-10 -- and s2 = 0 is the
      plain Normal exactly.
    """
    import pytensor.tensor as pt
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(3)
    group = np.repeat([0, 1, 2], [1, 3, 5])
    s2 = np.array([0.0, 2.5, 0.7])
    sigma = rng.uniform(0.5, 2.0, group.size)
    cov = np.diag(sigma**2) + s2[group][:, None] * (
        group[:, None] == group[None, :]
    )
    r = rng.multivariate_normal(np.zeros(group.size), cov)
    got = Instrument.shared_noise_logp(
        pt.as_tensor_variable(r), pt.as_tensor_variable(sigma), group, s2
    ).eval()
    want = multivariate_normal(np.zeros(group.size), cov).logpdf(r)
    assert got == pytest.approx(want, abs=1e-10)
    plain = Instrument.shared_noise_logp(
        pt.as_tensor_variable(r),
        pt.as_tensor_variable(sigma),
        group,
        np.zeros(3),
    ).eval()
    diag = -0.5 * np.sum(r**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
    assert plain == pytest.approx(diag, abs=1e-10)


@pytest.mark.parametrize(
    "rows, match",
    [
        (["0.5", "0.6", "0.5"], "share one value"),
        (["-0.5", "-0.5", "-0.5"], "must be >= 0"),
    ],
)
def test_err_shared_is_validated(chromatic, rows, match):
    """
    Given per-order rows of one exposure whose err_shared differ, or are
      negative,
    When prepared,
    Then it raises naming the rows: the common-mode error is one number
      per exposure.
    """
    tmp = chromatic["tmp"]
    t = chromatic["tags"][0]
    data = tmp / "shared_bad.rv"
    data.write_text(
        "".join(
            f"{t:.8f} 0.0 1.0 3600.0 {lam:.1f} {e}\n"
            for lam, e in zip(CHANNELS, rows)
        )
    )
    with pytest.raises(ValueError, match=match):
        _prepare(_entry(data, err_shared=5))


def test_err_shared_with_a_gp_raises(chromatic):
    """
    Given err_shared on a file that also asks for a GP,
    When prepared,
    Then it raises: each is its own likelihood for the file's rows.
    """
    entry = _entry(chromatic["data"], err_shared=5)
    entry["gp"] = "rotation"
    with pytest.raises(ValueError, match="cannot be\\s+combined"):
        _prepare(entry)


@pytest.mark.slow
def test_the_model_scores_err_shared_as_the_dense_likelihood(tmp_path):
    """
    Given a CORRECTED file (the dispatcher is the same for raw files, and a
      corrected build is cheap) of per-order rows in three exposures with
      a shared error per exposure,
    When the model is built,
    Then the file's rows leave the plain Normal and enter one Potential,
      and at the start point that Potential equals the dense multivariate
      normal of (data - model) with diag(err^2 + jitter^2) + err_shared^2
      inside each exposure -- the jitter read off the model, not assumed.
    """
    import pytensor
    from scipy.stats import multivariate_normal

    from exozippy.system import System

    rng = np.random.default_rng(5)
    t0 = 2460000.0
    rows = []
    for k, shared in enumerate([0.0, 3.0, 1.5]):
        for lam in CHANNELS:
            rows.append(
                (t0 + 3.1 * k, rng.normal(0.0, 5.0), 2.0 + k, lam, shared)
            )
    data = tmp_path / "corr.rv"
    np.savetxt(data, np.array(rows), fmt="%.6f")
    cfg = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [
            {
                "name": "X",
                "file": str(data),
                "columns": {
                    "time": 0,
                    "rv": 1,
                    "err": 2,
                    "channel": 3,
                    "err_shared": 4,
                },
            }
        ],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.03},
        "star.A.radius": {"initval": 1.0, "sigma": 0.03},
        "orbit.b.period": {"initval": 5.0},
        "orbit.b.tc": {"initval": t0},
    }
    system = System(cfg, user_params=params)
    system.prepare()
    model = system.build_model()
    names = [p.name for p in model.potentials]
    assert "rvinstrument.model.shared.X" in names
    assert "rvinstrument.model" not in [v.name for v in model.observed_RVs]
    rvi = system.rvinstrument
    start = system.get_raw_start(model)
    pot = model.potentials[names.index("rvinstrument.model.shared.X")]
    fn = pytensor.function(
        model.free_RVs,
        [pot, rvi._likelihood_mu_node, rvi.total_sigma(rvi.err)],
        on_unused_input="ignore",
    )
    lp, mu, sigma = fn(*[start[v.name] for v in model.free_RVs])
    resid = rvi.rv - np.asarray(mu)
    group = np.repeat([0, 1, 2], len(CHANNELS))
    s = np.array([0.0, 3.0, 1.5]) * rvi.units[0].to("solRad/d")
    cov = np.diag(np.asarray(sigma) ** 2) + (s[group] ** 2)[:, None] * (
        group[:, None] == group[None, :]
    )
    want = multivariate_normal(np.zeros(resid.size), cov).logpdf(resid)
    assert float(lp) == pytest.approx(want, rel=1e-10)
