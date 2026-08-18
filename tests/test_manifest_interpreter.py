"""One interpreter for the manifest vocabulary, shared by all its consumers.

`manifest.interpret_manifest_entry` is the single answer to "is this manifest
entry derived, which expressions: block does it select, and what options does
it carry".  Three places ask, at three lifecycle stages:

  graph.determine_pymc_build_order  (stage 4) -- build order
  Component.add_parameter           (stage 5) -- what actually gets built
  System.derived_params                       -- reporting / introspection

They each used to answer it themselves, and they disagreed: a dict without
"expr_key" ({"overrides": ...}, {"lower": ...}) is a FREE parameter to
add_parameter, but graph.py fell back to the "default" expression for any
dict.  That was latent until such a parameter gained an unused
`expressions.default` block in its defaults.yaml, at which point graph.py
wired edges to dependencies that were not in the manifest at all and the
build died with "Dependency Error" -- Band's linear-law u1 (whose Kipping
expression the manifest deliberately ignores) and planet.beam's "off" entry
both hit it.

They also agreed on something none of them should have: an expr_key naming
a block the resolved config does not define built the parameter FREE, in
silence.  That now raises (MissingExpressionError) -- see section 4, which
reproduces what the silence cost on a real derived parameter.

These tests pin the interpreter's contract, pin that graph.py reads entries
through it, and pin the real Band reproduction end to end.
"""

import numpy as np
import pytest

from exozippy.graph import determine_pymc_build_order
from exozippy.manifest import MissingExpressionError, interpret_manifest_entry
from exozippy.system import System

# ---------------------------------------------------------------------------
# 1. The interpreter's contract.
# ---------------------------------------------------------------------------


def test_none_entry_is_a_free_parameter():
    """
    Given a manifest entry of None,
    When it is interpreted,
    Then it names no expression and carries no options.
    """
    entry = interpret_manifest_entry(None)

    assert entry.names_expression is False
    assert entry.expr_key is None
    assert entry.options == {}


def test_bare_string_entry_names_that_expression():
    """
    Given a manifest entry that is the string "default",
    When it is interpreted,
    Then it names the "default" expression and carries no options.
    """
    entry = interpret_manifest_entry("default")

    assert entry.names_expression is True
    assert entry.expr_key == "default"
    assert entry.options == {}


def test_options_only_dict_is_a_free_parameter():
    """
    Given a dict manifest entry carrying only options (an "overrides" pin, a
      shape, a table note) and no "expr_key",
    When it is interpreted,
    Then it names NO expression, and every option survives.

    This is the rule the three consumers disagreed about.  Reading such an
    entry as derived-with-"default" is what produced spurious build-order
    edges (and, where the named deps were absent from the manifest, a hard
    Dependency Error).
    """
    raw = {
        "overrides": {"sigma": [0.0, np.nan]},
        "shape": (2,),
        "table_note": "pinned",
    }

    entry = interpret_manifest_entry(raw)

    assert entry.names_expression is False
    assert entry.expr_key is None
    assert entry.overrides == {"sigma": [0.0, np.nan]}
    assert entry.shape == (2,)
    assert entry.options["table_note"] == "pinned"


def test_dict_with_expr_key_is_derived_and_keeps_its_other_options():
    """
    Given a dict manifest entry carrying "expr_key" alongside other options,
    When it is interpreted,
    Then it names that expression and "expr_key" is stripped from the options.
    """
    entry = interpret_manifest_entry(
        {"expr_key": "from_sv", "force_node": True}
    )

    assert entry.expr_key == "from_sv"
    assert entry.options == {"force_node": True}
    assert "expr_key" not in entry.options


def test_interpreting_an_entry_does_not_mutate_the_manifest():
    """
    Given a dict manifest entry,
    When it is interpreted and the resulting options are mutated,
    Then the original entry is unchanged.

    add_parameter pops shape/names/overrides/deps off the options; the live
    manifest is read again at every stage and must survive that.
    """
    raw = {"expr_key": "default", "shape": (3,)}

    options = dict(interpret_manifest_entry(raw).options)
    options.pop("shape")

    assert raw == {"expr_key": "default", "shape": (3,)}


def test_expression_config_raises_when_the_named_block_is_absent():
    """
    Given an entry naming an expression the resolved config does not define,
    When expression_config is asked for it,
    Then MissingExpressionError names the component, the parameter, the
      missing key and the keys that ARE available, and says how to declare a
      free parameter instead.

    It used to answer None -- "free as built" -- so a typo, a renamed block
    or a deleted expressions: section silently turned a derived parameter
    into a sampled one.
    """
    entry = interpret_manifest_entry("default")

    with pytest.raises(MissingExpressionError) as exc:
        entry.expression_config({"other": {}}, where="compA.f_source")

    msg = str(exc.value)
    assert "compA.f_source" in msg
    assert "'default'" in msg
    assert "other" in msg  # the available keys
    assert "None" in msg and "expr_key" in msg  # the two free spellings

    with pytest.raises(MissingExpressionError, match=r"available: \(none\)"):
        entry.expression_config({})
    with pytest.raises(MissingExpressionError):
        entry.expression_config(None)


def test_expression_config_returns_the_block_it_names():
    """
    Given an entry naming an expression the config does define,
    When expression_config is asked for it,
    Then that block comes back, and an entry naming nothing still gets None.
    """
    block = {"func_name": "calc_x", "deps": ["y"]}

    assert (
        interpret_manifest_entry("default").expression_config(
            {"default": block}
        )
        is block
    )
    assert interpret_manifest_entry(None).expression_config({}) is None
    assert (
        interpret_manifest_entry({"shape": (3,)}).expression_config({}) is None
    )


def test_manifest_deps_override_the_expression_block_deps():
    """
    Given an entry that declares its own "deps" (per-instance dependencies,
      as orbit's body groups do),
    When dep_names is asked for,
    Then the entry's list wins over the expression block's.
    """
    entry = interpret_manifest_entry(
        {"expr_key": "default", "deps": ["star.0.mass"]}
    )

    assert entry.dep_names({"deps": ["m_total"]}) == ["star.0.mass"]
    assert interpret_manifest_entry("default").dep_names(
        {"deps": ["m_total"]}
    ) == ["m_total"]


def test_empty_string_expr_key_is_normalized_to_no_expression():
    """
    Given an entry whose expr_key is the empty string,
    When it is interpreted,
    Then it names no expression.

    add_parameter tested `if expr_key and ...` (falsy -> free) while
    derived_params tested `is not None` (-> derived).  The near-miss is
    exactly the sort of thing three hand-written readers drift apart on.
    """
    assert interpret_manifest_entry("").names_expression is False
    assert interpret_manifest_entry({"expr_key": ""}).names_expression is False


def test_unparseable_entry_raises_naming_the_type():
    """
    Given a manifest entry that is neither None, a string, nor a dict,
    When it is interpreted,
    Then a TypeError names the offending type rather than failing later
      with a cryptic dict() or membership error.
    """
    with pytest.raises(TypeError, match="int"):
        interpret_manifest_entry(7)


# ---------------------------------------------------------------------------
# 2. graph.py reads entries through the interpreter.
#
# The regression: an options-only dict entry on a parameter whose
# defaults.yaml DOES define expressions.default, whose deps are not in the
# manifest.  Pre-fix graph.py fell back to "default" and raised
# "Dependency Error: compA.u1 depends on compA.q1".
# ---------------------------------------------------------------------------


class _FakeComp:
    def __init__(self, prefix, manifest):
        self.prefix = prefix
        self.manifest = manifest
        self.n_elements = 1


class _FakeConfigManager:
    """Resolves every parameter to the same unused expressions block."""

    def __init__(self, expressions_map):
        self._map = expressions_map

    def resolve(self, prefix, param_name, shape=None, names=None):
        expressions = self._map.get(f"{prefix}.{param_name}")
        return {"expressions": expressions} if expressions else {}


def test_options_only_entry_with_unused_expression_adds_no_edges():
    """
    Given a free parameter whose manifest entry is an options-only dict, and
      whose defaults.yaml carries an expressions.default block naming deps
      that are NOT in the manifest (Band's linear-law u1),
    When the build order is determined,
    Then the parameter is ordered as a free node instead of raising a
      Dependency Error for the unused expression's deps.
    """
    comps = {
        "compA": _FakeComp("compA", {"u1": {"overrides": {"sigma": [0.0]}}})
    }
    cm = _FakeConfigManager({"compA.u1": {"default": {"deps": ["q1", "q2"]}}})

    order = determine_pymc_build_order(comps, cm)

    assert order == ["compA.u1"]


def test_explicit_expr_key_still_produces_the_edge():
    """
    Given the same defaults.yaml block, but a manifest entry that explicitly
      names it,
    When the build order is determined,
    Then the dependency is ordered before the derived parameter.

    The interpreter may only ever REMOVE edges add_parameter does not use;
    it must never drop one it does.
    """
    comps = {
        "compA": _FakeComp(
            "compA", {"q1": None, "u1": {"expr_key": "default"}}
        )
    }
    cm = _FakeConfigManager({"compA.u1": {"default": {"deps": ["q1"]}}})

    order = determine_pymc_build_order(comps, cm)

    assert order.index("compA.q1") < order.index("compA.u1")


def test_graph_raises_on_an_expr_key_no_expression_block_defines():
    """
    Given a manifest entry naming an expressions: block the resolved config
      does not define,
    When the build order is determined,
    Then MissingExpressionError names the component, the parameter, the
      missing key and the keys that ARE available.

    graph.py used to skip such an entry and add no edge, in lockstep with
    add_parameter building the parameter FREE -- consistent, and silently
    wrong for a parameter meant to be derived.
    """
    comps = {"compA": _FakeComp("compA", {"u1": "kipping"})}
    cm = _FakeConfigManager({"compA.u1": {"default": {"deps": ["q1", "q2"]}}})

    with pytest.raises(MissingExpressionError) as exc:
        determine_pymc_build_order(comps, cm)

    msg = str(exc.value)
    assert "compA.u1" in msg
    assert "'kipping'" in msg
    assert "default" in msg


def test_every_manifest_consumer_reads_through_the_interpreter():
    """
    Given the three modules that consume the manifest vocabulary,
    When their source is inspected,
    Then each imports interpret_manifest_entry and none of them re-derives
      the expr_key rule itself.

    The behavioural disagreement was fixed by hand once already; this pins
    the structural half, which is what let it happen.
    """
    import inspect

    import exozippy.components.component as component_mod
    import exozippy.graph as graph_mod
    import exozippy.system as system_mod

    for mod in (graph_mod, component_mod, system_mod):
        src = inspect.getsource(mod)
        assert "interpret_manifest_entry" in src, mod.__name__
        # The hand-rolled forms: pulling "expr_key" out of a raw entry, or
        # special-casing a bare-string entry, anywhere but the interpreter.
        code = "\n".join(
            line
            for line in src.splitlines()
            if not line.strip().startswith("#")
        )
        assert 'pop("expr_key"' not in code, mod.__name__
        assert 'get("expr_key"' not in code, mod.__name__


# ---------------------------------------------------------------------------
# 3. The real reproduction: Band's linear-law u1.
#
# Band pins the LD of any band no consumer reads, through the manifest
# "overrides" channel -- so with two bands and one consumer, the linear law's
# manifest["u1"] becomes an options-only dict, while band/defaults.yaml still
# carries u1's (deliberately unused) Kipping expression with deps q1/q2.
# Under the old graph.py this config could not be built at all.
# ---------------------------------------------------------------------------

T0 = 2460025.0
TE = 30.0
U0 = 0.1


@pytest.fixture(scope="module")
def unread_linear_band_system(tmp_path_factory):
    """Given a finite-source PSPL fit with two linear-law bands, only one of
    which any component reads, when the system is prepared, provide it."""
    path = tmp_path_factory.mktemp("manifest_interp") / "lc.dat"
    t = np.linspace(T0 - 2 * TE, T0 + 2 * TE, 60)
    u = np.sqrt(U0**2 + ((t - T0) / TE) ** 2)
    amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    np.savetxt(
        path,
        np.column_stack([t, 18.0 - 2.5 * np.log10(amp), np.full(60, 0.01)]),
    )

    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [
            {
                "name": "Lens",
                "lens_ndx": 0,
                "source_ndx": 1,
                "finite_source": True,
                "t0_par": T0,
                "use_op": False,
                "mmexofast": False,
            }
        ],
        "mulensinstrument": [
            {"name": "OGLE", "file": str(path), "filter": "I", "band": "I"}
        ],
        # "I" is read (finite source); "V" is read by nothing at all.
        "band": [
            {"name": "I", "filter": "I", "ld_law": "linear"},
            {"name": "V", "filter": "V", "ld_law": "linear"},
        ],
    }
    params = {
        "lens.Lens.t_0": {"initval": T0},
        "lens.Lens.u_0": {"initval": U0},
        "lens.Lens.t_E": {"initval": TE},
        "lens.Lens.rho": {"initval": 1.0e-3},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("Lens", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}

    system = System(config, user_params=params)
    system.prepare()
    return system


def test_band_linear_u1_is_an_options_only_entry(unread_linear_band_system):
    """
    Given a linear-law band set in which one band's LD is unread,
    When registration has run,
    Then band.u1's manifest entry is an options-only dict -- the shape the
      two readers disagreed about -- while u1's defaults.yaml expression
      block (Kipping, deps q1/q2) is still there and still unused.
    """
    band = unread_linear_band_system.active_components["band"]
    entry = interpret_manifest_entry(band.manifest["u1"])

    assert isinstance(band.manifest["u1"], dict)
    assert entry.names_expression is False
    assert "overrides" in entry.options
    assert "q1" not in band.manifest and "q2" not in band.manifest

    cfg = unread_linear_band_system.config_manager.resolve(
        "band", "u1", shape=(band.n_elements,)
    )
    assert cfg["expressions"]["default"]["deps"] == ["q1", "q2"]


def test_unread_linear_band_builds_with_u1_free(unread_linear_band_system):
    """
    Given that same system,
    When the build order is determined and the model is built,
    Then band.u1 is ordered as a free parameter, and neither the sort nor
      the build asks for the q1/q2 the unused expression names.

    Pre-fix this raised "Dependency Error: band.u1 depends on band.q1,
    which is not defined in any manifest."
    """
    system = unread_linear_band_system

    order = determine_pymc_build_order(
        system.active_components, system.config_manager
    )

    assert "band.u1" in order
    assert "band.q1" not in order
    assert ("band", "u1") not in system.derived_params()

    model = system.build_model()
    assert model is not None


# ---------------------------------------------------------------------------
# 4. The cost of the silent fallback: mulensinstrument.f_source.
#
# f_source is derived from log_f_total and q_source.  Break its expr_key and
# the old readers agreed the parameter was FREE: examples/ob08092 grew an
# f_source_raw in model.free_RVs and its start logp moved from +6187.7 to
# -6.46e9, with no error and no warning anywhere.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pspl_system(tmp_path_factory):
    """Given a point-source point-lens fit with one light curve, when it is
    prepared, provide the system."""
    path = tmp_path_factory.mktemp("expr_key") / "lc.dat"
    t = np.linspace(T0 - 2 * TE, T0 + 2 * TE, 60)
    u = np.sqrt(U0**2 + ((t - T0) / TE) ** 2)
    amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    np.savetxt(
        path,
        np.column_stack([t, 18.0 - 2.5 * np.log10(amp), np.full(60, 0.01)]),
    )

    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [
            {
                "name": "Lens",
                "lens_ndx": 0,
                "source_ndx": 1,
                "t0_par": T0,
                "use_op": False,
                "mmexofast": False,
            }
        ],
        "mulensinstrument": [
            {"name": "OGLE", "file": str(path), "filter": "I"}
        ],
    }
    params = {
        "lens.Lens.t_0": {"initval": T0},
        "lens.Lens.u_0": {"initval": U0},
        "lens.Lens.t_E": {"initval": TE},
    }
    for nm in ("Lens", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}

    system = System(config, user_params=params)
    system.prepare()
    return system


def test_f_source_builds_derived_when_its_expr_key_resolves(pspl_system):
    """
    Given the intact mulensinstrument manifest,
    When the system is built,
    Then f_source is reported derived and is not a sampled variable.
    """
    system = pspl_system

    model = system.build_model()

    assert ("mulensinstrument", "f_source") in system.derived_params()
    assert not [rv for rv in model.free_RVs if "f_source" in rv.name], (
        "f_source must be derived from log_f_total and q_source"
    )


def test_broken_f_source_expr_key_raises_instead_of_sampling_it(pspl_system):
    """
    Given that same system with f_source's expr_key pointed at a block
      mulensing/defaults.yaml does not define,
    When the build order is determined, or derived_params is asked,
    Then MissingExpressionError names mulensinstrument.f_source, the missing
      key and the available ones -- instead of building a sampled f_source.
    """
    system = pspl_system
    comp = system.active_components["mulensinstrument"]
    original = comp.manifest["f_source"]
    comp.manifest["f_source"] = "defualt"  # the realistic failure: a typo

    try:
        # ACT / ASSERT -- the build path
        with pytest.raises(MissingExpressionError) as exc:
            determine_pymc_build_order(
                system.active_components, system.config_manager
            )
        msg = str(exc.value)
        assert "mulensinstrument.f_source" in msg
        assert "'defualt'" in msg
        assert "available: default" in msg

        # ACT / ASSERT -- the reporting path, which never builds a model and
        # so would otherwise keep answering "derived" for a parameter that
        # cannot be built at all.
        with pytest.raises(MissingExpressionError):
            system.derived_params()
    finally:
        comp.manifest["f_source"] = original
