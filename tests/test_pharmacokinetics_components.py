"""The pharmacokinetics components (subject + assay).

See ``src/exozippy/components/pharmacokinetics/README.md``: written by an
astrophysicist and an LLM with no domain reviewer. These tests pin the wiring --
config parsing, the subject map, unit conversion, and that a model builds and
is differentiable -- not that the modelling choices are right.

Everything here runs on SYNTHETIC data generated in ``tmp_path``. The real
Theophylline example fetches its data over the network, which a test must not
depend on.
"""

import numpy as np
import pytest

from exozippy import reporting
from exozippy.components.factory import discover_components
from exozippy.components.pharmacokinetics.assay import Assay
from exozippy.components.pharmacokinetics.subject import Subject
from exozippy.system import System

# name -> (weight kg, dose mg/kg, CL L/hr, V L, ka 1/hr)
TRUTH = {
    "S1": (79.6, 4.02, 2.8, 32.0, 1.5),
    "S2": (72.4, 4.40, 3.4, 29.0, 1.2),
    "S3": (70.5, 4.53, 2.2, 35.0, 1.9),
}
TIMES = np.array([0.25, 0.57, 1.12, 2.02, 3.82, 5.1, 7.03, 9.05, 12.12, 24.37])


def _curve(t, dose, cl, v, ka):
    ke = cl / v
    return (dose * ka) / (v * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


@pytest.fixture
def synth_csv(tmp_path):
    """A 3-subject assay file, subjects labelled with BARE NUMBERS.

    Bare numbers because that is how clinical data label subjects, and it is
    what forces the ``subject_prefix`` question the core's ban on numeric
    instance names creates.
    """
    rng = np.random.default_rng(7)
    rows = ["subject,time,conc"]
    for name, (wt, dose_per_kg, cl, v, ka) in TRUTH.items():
        c = _curve(TIMES, dose_per_kg * wt, cl, v, ka)
        obs = c + rng.normal(0.0, np.sqrt(0.15**2 + (0.08 * c) ** 2))
        for t, y in zip(TIMES, obs):
            rows.append(f"{name[1:]},{t},{y:.6f}")
    path = tmp_path / "synth.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


def _config(csv, prefix_dir, **assay_overrides):
    assay = {
        "name": "synth",
        "datafile": str(csv),
        "columns": {"subject": "subject", "time": "time", "obs": "conc"},
        "subject_prefix": "S",
        "conc_unit": "mg/L",
        "time_unit": "hr",
    }
    assay.update(assay_overrides)
    return {
        "name": "pk_test",
        "prefix": str(prefix_dir / "pk"),
        "subject": [
            {"name": n, "weight": wt, "dose": d, "dose_unit": "mg/kg"}
            for n, (wt, d, *_r) in TRUTH.items()
        ],
        "assay": [assay],
    }


@pytest.fixture
def prepared(synth_csv, tmp_path):
    system = System(_config(synth_csv, tmp_path), user_params={})
    system.prepare()
    return system


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_both_components_are_auto_discovered():
    """Given the factory sweep, Then subject and assay are registered.

    No registration step exists by design (components.md), so this pins that
    a component in a new directory is found the way the contract promises.
    """
    registry = discover_components()
    assert registry["subject"] is Subject
    assert registry["assay"] is Assay


def test_subject_declares_that_it_expects_suppressed_modes():
    """Given the flip-flop degeneracy, Then the component opts in generically.

    ``expects_suppressed_modes`` is declared on ``Component`` precisely so a
    component with degenerate solutions can turn on hot-chain retention
    without anyone editing the sampler layer. A pharmacology component using
    it is the cleanest available evidence that the hook is component-agnostic.
    """
    assert Subject.expects_suppressed_modes is True


# ---------------------------------------------------------------------------
# Dose: the unit carries the per-weight distinction
# ---------------------------------------------------------------------------


def test_a_per_weight_dose_is_multiplied_by_the_weight(prepared):
    """Given dose_unit mg/kg, Then the absolute dose is dose * weight."""
    expected = [wt * d for wt, d, *_r in TRUTH.values()]
    assert prepared.subject.dose_mg == pytest.approx(expected)


def test_an_absolute_dose_is_taken_as_written(synth_csv, tmp_path):
    """Given dose_unit mg, Then the weight is irrelevant to the dose."""
    cfg = _config(synth_csv, tmp_path)
    for entry in cfg["subject"]:
        entry["dose"] = 320.0
        entry["dose_unit"] = "mg"
    system = System(cfg, user_params={})
    system.prepare()
    assert prepared_dose(system) == pytest.approx([320.0] * 3)


def prepared_dose(system):
    return list(system.subject.dose_mg)


def test_a_gram_dose_converts(synth_csv, tmp_path):
    """Given dose_unit g, Then it becomes mg.

    The conversion goes through astropy rather than a hand-written factor,
    which is the rule this codebase learned the expensive way.
    """
    cfg = _config(synth_csv, tmp_path)
    for entry in cfg["subject"]:
        entry["dose"] = 0.32
        entry["dose_unit"] = "g"
    system = System(cfg, user_params={})
    system.prepare()
    assert prepared_dose(system) == pytest.approx([320.0] * 3)


def test_a_per_weight_dose_without_a_weight_raises(synth_csv, tmp_path):
    """Given mg/kg and no weight, Then it raises rather than under-dosing 70x.

    Silently treating a per-kg dose as absolute would make every concentration
    prediction wrong by the body weight -- a large, plausible-looking error.
    """
    cfg = _config(synth_csv, tmp_path)
    cfg["subject"][0].pop("weight")
    with pytest.raises(ValueError, match="weight"):
        System(cfg, user_params={}).prepare()


def test_an_uninterpretable_dose_unit_raises(synth_csv, tmp_path):
    """Given a dose_unit that is neither amount nor amount-per-weight, Then raise."""
    cfg = _config(synth_csv, tmp_path)
    cfg["subject"][0]["dose_unit"] = "L"
    with pytest.raises(ValueError, match="neither an amount"):
        System(cfg, user_params={}).prepare()


# ---------------------------------------------------------------------------
# The subject map
# ---------------------------------------------------------------------------


def test_the_subject_map_pairs_each_row_with_its_subject(prepared):
    """Given the data, Then subject_map indexes the declared subjects.

    Built from NAMES rather than row order, so a file whose subjects are
    interleaved or out of order still pairs correctly.
    """
    smap = prepared.assay.subject_map
    assert smap.shape == (len(TRUTH) * TIMES.size,)
    expected = np.repeat(np.arange(len(TRUTH)), TIMES.size)
    assert np.array_equal(smap, expected)


def test_interleaved_rows_still_pair_correctly(synth_csv, tmp_path):
    """Given shuffled rows, Then each observation keeps its own subject.

    Pins that the map comes from the subject COLUMN and not from row order --
    the failure that would otherwise be invisible, since a shuffled file has
    the same rows and the same likelihood shape.
    """
    lines = synth_csv.read_text().splitlines()
    header, body = lines[0], lines[1:]
    rng = np.random.default_rng(3)
    order = rng.permutation(len(body))
    shuffled = tmp_path / "shuffled.csv"
    shuffled.write_text("\n".join([header] + [body[i] for i in order]) + "\n")

    system = System(_config(shuffled, tmp_path), user_params={})
    system.prepare()

    names = [str(body[i].split(",")[0]) for i in order]
    expected = [list(TRUTH).index("S" + n) for n in names]
    assert np.array_equal(system.assay.subject_map, np.asarray(expected))


def test_a_subject_in_the_data_with_no_block_raises(synth_csv, tmp_path):
    """Given an undeclared subject, Then it raises naming the numeric-name trap.

    Dropping those rows would fit a subset of the data and report it as the
    whole. The message names ``subject_prefix`` because a numerically labelled
    file is the overwhelmingly likely cause.
    """
    cfg = _config(synth_csv, tmp_path)
    cfg["subject"] = cfg["subject"][:2]
    with pytest.raises(ValueError, match="subject_prefix"):
        System(cfg, user_params={}).prepare()


def test_a_subject_with_no_observations_warns_but_builds(
    synth_csv, tmp_path, caplog
):
    """Given a declared subject absent from the data, Then warn and continue.

    Legitimate -- a dropout, or a subject whose samples are in another file --
    and its parameters are then prior-only, which is what the warning says.
    """
    cfg = _config(synth_csv, tmp_path)
    cfg["subject"].append(
        {"name": "S9", "weight": 70.0, "dose": 4.0, "dose_unit": "mg/kg"}
    )
    system = System(cfg, user_params={})
    with caplog.at_level("WARNING"):
        system.prepare()
    assert "no observations" in caplog.text
    assert system.subject.n_elements == 4


# ---------------------------------------------------------------------------
# Data-side unit conversion
# ---------------------------------------------------------------------------


def test_ng_per_ml_and_mg_per_l_describe_the_same_data(synth_csv, tmp_path):
    """Given the same data in ng/mL, Then the model logp is unchanged.

    mg/L -> ng/mL is exactly 1000. No Parameter owns a data column, so this
    conversion is the component's own; getting it wrong is a factor-of-1000
    error that every fitted concentration would silently absorb.
    """
    scaled = tmp_path / "scaled.csv"
    lines = synth_csv.read_text().splitlines()
    out = [lines[0]]
    for row in lines[1:]:
        subj, t, c = row.split(",")
        out.append(f"{subj},{t},{float(c) * 1000.0:.6f}")
    scaled.write_text("\n".join(out) + "\n")

    base = System(_config(synth_csv, tmp_path), user_params={})
    base.prepare()
    other = System(
        _config(scaled, tmp_path, conc_unit="ng/mL"), user_params={}
    )
    other.prepare()

    assert other.assay.all_obs == pytest.approx(base.assay.all_obs, rel=1e-9)


def test_minutes_and_hours_describe_the_same_data(synth_csv, tmp_path):
    """Given times in minutes, Then they are converted to the internal hours."""
    scaled = tmp_path / "minutes.csv"
    lines = synth_csv.read_text().splitlines()
    out = [lines[0]]
    for row in lines[1:]:
        subj, t, c = row.split(",")
        out.append(f"{subj},{float(t) * 60.0:.6f},{c}")
    scaled.write_text("\n".join(out) + "\n")

    system = System(_config(scaled, tmp_path, time_unit="min"), user_params={})
    system.prepare()
    assert system.assay.all_time == pytest.approx(
        np.tile(TIMES, len(TRUTH)), rel=1e-9
    )


def test_a_non_convertible_concentration_unit_raises(synth_csv, tmp_path):
    """Given conc_unit 'hr', Then it raises rather than scaling by nonsense."""
    with pytest.raises(ValueError, match="not convertible"):
        System(
            _config(synth_csv, tmp_path, conc_unit="hr"), user_params={}
        ).prepare()


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_columns_may_be_given_by_index(synth_csv, tmp_path):
    """Given integer column indices, Then they resolve the same as names."""
    by_index = System(
        _config(
            synth_csv, tmp_path, columns={"subject": 0, "time": 1, "obs": 2}
        ),
        user_params={},
    )
    by_index.prepare()
    by_name = System(_config(synth_csv, tmp_path), user_params={})
    by_name.prepare()
    assert by_index.assay.all_obs == pytest.approx(by_name.assay.all_obs)


def test_an_unknown_column_name_raises_listing_the_available_ones(
    synth_csv, tmp_path
):
    """Given a misspelled column, Then the error lists what the file has."""
    with pytest.raises(ValueError, match="Available"):
        System(
            _config(synth_csv, tmp_path, columns={"obs": "concentration"}),
            user_params={},
        ).prepare()


def test_two_roles_on_one_column_raises(synth_csv, tmp_path):
    """Given time and obs mapped to one column, Then it raises.

    A config that reads the concentration column as the time would produce a
    finite, entirely meaningless fit.
    """
    with pytest.raises(ValueError, match="two roles"):
        System(
            _config(
                synth_csv,
                tmp_path,
                columns={"subject": 0, "time": 2, "obs": 2},
            ),
            user_params={},
        ).prepare()


def test_a_missing_datafile_raises_pointing_at_the_fetcher(tmp_path):
    """Given no data file, Then the error names the fetch utility.

    The example's data are not redistributed, so "file not found" is the
    expected first experience and it should say what to do about it.
    """
    cfg = _config(tmp_path / "absent.csv", tmp_path)
    with pytest.raises(FileNotFoundError, match="exozippy-fetch-theoph"):
        System(cfg, user_params={}).prepare()


# ---------------------------------------------------------------------------
# The built model
# ---------------------------------------------------------------------------


def test_the_model_builds_with_a_finite_logp_and_gradient(prepared):
    """Given a prepared system, When built, Then logp and dlogp are finite.

    The gradient half is not redundant: the ka == ke floor exists so that the
    gradient survives, and a NaN in one term poisons the whole vector.
    """
    model = prepared.build_model()
    point = model.initial_point()

    logp = float(model.compile_logp()(point))
    assert np.isfinite(logp)
    assert np.all(np.isfinite(model.compile_dlogp()(point)))


def test_the_sampled_coordinates_are_the_logs(prepared):
    """Given the built model, Then CL, V and ka are sampled in log10.

    The physical values are derived, mirroring star.logmass and mulensing's
    log_s. A change that sampled the linear values would still fit and would
    quietly alter the prior.
    """
    model = prepared.build_model()
    names = {v.name for v in model.free_RVs}
    assert names == {
        "subject.log_cl_raw",
        "subject.log_v_raw",
        "subject.log_ka_raw",
        "assay.sigma_add_raw",
        "assay.sigma_prop_raw",
    }


def test_no_per_observation_deterministic_is_stored(prepared):
    """Given the built model, Then nothing named per-observation is tracked.

    A Deterministic over the predicted concentrations would be n_obs x n_draws
    x n_chains floats in the trace, and every entry would be treated as a
    parameter by the corner plot and the trace pages.  That is not a
    theoretical cost: it got the Theophylline example's wrap-up OOM-killed
    after the fit had finished.  The curve is a deterministic function of the
    parameters, so a plotter recomputes it instead.
    """
    model = prepared.build_model()
    n_obs = prepared.assay.all_obs.size

    for var in model.deterministics:
        size = int(np.prod(var.shape.eval()))
        assert size < n_obs, (
            f"{var.name} has {size} entries for {n_obs} observations -- a "
            f"per-observation Deterministic bloats the trace and floods the "
            f"corner plot"
        )


def test_building_twice_gives_an_independent_graph(prepared):
    """Given one prepared System, When built twice, Then both models work.

    ``System.build_model`` may be called more than once (the GUI does), and
    the second model must not be handed the first one's nodes.
    """
    first = prepared.build_model()
    logp_first = float(first.compile_logp()(first.initial_point()))

    second = prepared.build_model()
    logp_second = float(second.compile_logp()(second.initial_point()))

    assert logp_second == pytest.approx(logp_first)


def test_derived_quantities_are_reported_but_not_sampled(prepared):
    """Given the manifest, Then ke/t_half/tmax/cmax/auc are derived.

    They are what a reader of a PK table actually wants, and none of them is
    a free dimension.
    """
    # derived_params() keys are (component, parameter) TUPLES, not dotted
    # strings -- worth pinning by construction rather than by spelling, since
    # a string membership test against a set of tuples is quietly always False.
    derived = prepared.derived_params()
    for name in ("ke", "t_half", "tmax", "cmax", "auc", "cl", "v", "ka"):
        assert ("subject", name) in derived, name
    assert ("subject", "log_cl") not in derived


def test_the_dose_is_pinned_to_the_configured_value(prepared):
    """Given a dose, Then it is a fixed parameter, not a fitted one.

    It appears in the table because a reader checking a PK fit wants to see
    the dose it assumed, but it must never acquire a raw coordinate.
    """
    model = prepared.build_model()
    assert "subject.dose_raw" not in {v.name for v in model.free_RVs}
    assert prepared.subject.dose.initval == pytest.approx(
        [wt * d for wt, d, *_r in TRUTH.values()]
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_example_asks_for_the_pharmacometrics_convention():
    """Given the shipped example config, Then it reports at exactly 95%.

    Not 68.3% (astronomy's 1 sigma, this code's default) and not 95.45%
    (2 sigma). The distinction is the whole reason the setting exists, so the
    example is pinned rather than trusted to stay right.
    """
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "examples" / "theophylline" / "theophylline.yaml").read_text()
    )

    width = config["reporting"]["credible_interval"]
    assert width == 0.95
    assert reporting.sigma_multiple(width) is None
    assert reporting.label(width) == "95"
