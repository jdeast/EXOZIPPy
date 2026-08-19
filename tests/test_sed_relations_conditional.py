"""The SED relations only apply where there is an SED (review 1.9.4).

``sed/symbolic_physics.py`` sets ``comp_key = "star"`` so its two
defaulting relations instantiate once per star -- but what they describe
(teffsed, radiussed) only exists when there is a ``sed:`` block, which is
exactly the condition ``Star.register_parameters`` already uses to decide
whether to declare them.
"""

from exozippy.config import ConfigManager

# ---------------------------------------------------------------------------
# 1.9.4 -- the SED relations only apply where there is an SED
# ---------------------------------------------------------------------------

_STAR_ONLY = {"star": [{"name": "A"}]}
_STAR_AND_SED = {"star": [{"name": "A"}], "sed": {"file": "unused.sed"}}


def _sed_symbols(system_config):
    cm = ConfigManager({}, system_config=system_config)
    return {
        path
        for path in cm.master_symbol_map
        if path.endswith(".teffsed") or path.endswith(".radiussed")
    }


def _relations_mentioning_sed(system_config):
    cm = ConfigManager({}, system_config=system_config)
    return [
        rel
        for rel in cm.all_relations
        if any(
            s.name.endswith(".teffsed") or s.name.endswith(".radiussed")
            for s in rel.free_symbols
        )
    ]


def test_a_config_with_no_sed_block_gets_no_sed_symbols():
    """
    Given a system with stars and no sed: block,
    When the ConfigManager loads the relation files,
    Then no teffsed/radiussed symbol and no relation naming one is
      registered.

    Regression: sed/symbolic_physics.py sets comp_key = "star" so its two
    relations instantiate once per star; registered unconditionally they
    put two phantom leaf symbols per star -- each with a default-armor
    value, a provenance-ledger row and an inject-back initval -- into
    every non-SED config, for parameters Star.register_parameters does
    not even declare there.
    """
    # ARRANGE / ACT
    symbols = _sed_symbols(_STAR_ONLY)
    relations = _relations_mentioning_sed(_STAR_ONLY)

    # ASSERT
    assert symbols == set()
    assert relations == []


def test_a_config_with_an_sed_block_still_gets_them():
    """
    Given the same system plus a sed: block,
    When the ConfigManager loads the relation files,
    Then teffsed and radiussed are registered and their defaulting
      relations are present -- the conditioning must not disable the
      feature it is guarding.
    """
    # ARRANGE / ACT
    symbols = _sed_symbols(_STAR_AND_SED)
    relations = _relations_mentioning_sed(_STAR_AND_SED)

    # ASSERT
    assert symbols == {"star.0.teffsed", "star.0.radiussed"}
    assert len(relations) == 2
