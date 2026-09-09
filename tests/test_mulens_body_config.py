"""Known-key enforcement for the post-split mulens config blocks (R3).

Ruling R3 (8.6.17) makes the config split a hard break with no
compatibility path.  What R3 does NOT license is silence: the migration
mistake a real user makes is keeping the old event-level options on a
`lens:` entry while adding the new `body:` key -- pre-fix, that built
without a diagnostic, with finite_source=False and every fit* flag
ignored.  These tests pin the gate in bodies.py: an unknown or misplaced
key is refused with a message naming the block it belongs on now.

The key sets are also pinned against each component's config_schema() so a
new schema key cannot silently diverge from the gate.
"""

import pytest

from exozippy.components.mulensing import bodies
from exozippy.components.mulensing.lens import Lens
from exozippy.components.mulensing.mulensevent import MulensEvent
from exozippy.components.mulensing.source import Source


class _DummyConfigManager:
    def __init__(self, system_config=None):
        self.system_config = system_config or {}
        self.user_params = {}


def _system_config(event=None, lens=None, source=None):
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "b"}],
        "mulensevent": event if event is not None else [{}],
        "lens": lens
        if lens is not None
        else [{"body": "star.Lens"}, {"body": "planet.b"}],
        "source": source if source is not None else [{"body": "star.Source"}],
    }


# ---------------------------------------------------------------------------
# The gate refuses misplaced keys, naming the right block
# ---------------------------------------------------------------------------


def test_event_options_on_a_lens_entry_with_body_are_refused():
    """
    Given a lens entry that keeps the old event-level options but adds the
      new `body:` key (the realistic half-migrated config),
    When the lens block is parsed,
    Then a ValueError names each misplaced key and the mulensevent block --
      instead of building with finite_source=False and the flags ignored,
      which is what the pre-fix code did.
    """
    cfg = _system_config(
        lens=[
            {
                "body": "star.Lens",
                "finite_source": True,
                "fitpirel": True,
                "t0_par": 2455379.571,
            },
            {"body": "planet.b"},
        ]
    )
    with pytest.raises(ValueError) as exc:
        Lens(cfg["lens"], _DummyConfigManager(cfg))
    msg = str(exc.value)
    assert "lens.0" in msg
    for key in ("finite_source", "fitpirel", "t0_par"):
        assert key in msg
    assert "mulensevent" in msg


def test_per_source_flags_on_the_mulensevent_block_are_refused():
    """
    Given star_constrains_rho and fitu0te on the mulensevent block (they
      moved to per-source entries in the split),
    When the event component is constructed,
    Then a ValueError names the source entry as their home -- pre-fix they
      were silently ignored.
    """
    event = [
        {"finite_source": True, "star_constrains_rho": False, "fitu0te": True}
    ]
    cfg = _system_config(event=event)
    with pytest.raises(ValueError) as exc:
        MulensEvent(event, _DummyConfigManager(cfg))
    msg = str(exc.value)
    assert "mulensevent.0" in msg
    assert "star_constrains_rho" in msg
    assert "fitu0te" in msg
    assert "'source:' entry" in msg


def test_companion_keys_on_a_source_entry_are_refused():
    """
    Given orbital_motion/orbit on a source entry,
    When the source block is parsed,
    Then the refusal points at the lens companion entry (the per-companion
      home) -- xallarap spells it source_orbital_motion on mulensevent.
    """
    cfg = _system_config(
        source=[{"body": "star.Source", "orbital_motion": "linear"}]
    )
    with pytest.raises(ValueError) as exc:
        Source(cfg["source"], _DummyConfigManager(cfg))
    msg = str(exc.value)
    assert "source.0" in msg
    assert "orbital_motion" in msg
    assert "'lens:' entry" in msg


@pytest.mark.parametrize(
    "block, ctor",
    [
        ("mulensevent", "event"),
        ("lens", "lens"),
        ("source", "source"),
    ],
)
def test_arbitrary_garbage_keys_are_refused_on_every_block(block, ctor):
    """
    Given a made-up key on each of the three blocks,
    When the block is parsed,
    Then it is refused (pre-fix: accepted silently), and the message lists
      the keys the block does accept.
    """
    cfg = _system_config()
    entry = dict(cfg[block][0])
    entry["definitely_not_a_key"] = 1
    cfg[block][0] = entry

    with pytest.raises(ValueError) as exc:
        if ctor == "event":
            MulensEvent(cfg["mulensevent"], _DummyConfigManager(cfg))
        elif ctor == "lens":
            Lens(cfg["lens"], _DummyConfigManager(cfg))
        else:
            Source(cfg["source"], _DummyConfigManager(cfg))
    msg = str(exc.value)
    assert "definitely_not_a_key" in msg
    assert "accepts" in msg


def test_presplit_spellings_are_named_as_presplit_even_with_body_present():
    """
    Given a lens entry that carries both `body:` and the dead pre-v0.1.0
      `lenses:` key,
    When the block is parsed,
    Then the refusal says the key is a pre-v0.1.0 spelling rather than a
      generic unknown.
    """
    cfg = _system_config(
        lens=[
            {"body": "star.Lens", "lenses": ["star.0"]},
            {"body": "planet.b"},
        ]
    )
    with pytest.raises(ValueError, match="pre-v0.1.0"):
        Lens(cfg["lens"], _DummyConfigManager(cfg))


# ---------------------------------------------------------------------------
# The gate does not over-refuse (control the other way)
# ---------------------------------------------------------------------------


def test_every_schema_key_still_builds():
    """
    Given blocks using their full legal vocabularies,
    When the components are constructed,
    Then no key is refused -- the gate must not reject the documented
      config surface.
    """
    event = [
        {
            "name": "ev",
            "finite_source": True,
            "t0_par": 2455379.5,
            "backend": "vbm_direct",
            "mag_method": "auto_vbbl",
            "use_op": False,
            "mmexofast": False,
            "fitmurel": False,
            "fitpirel": False,
            "fitthetae": False,
        }
    ]
    lens = [
        {"body": "star.Lens"},
        {"body": "planet.b", "orbital_motion": "linear"},
    ]
    source = [
        {
            "body": "star.Source",
            "fitu0te": True,
            "star_constrains_rho": False,
        }
    ]
    cfg = _system_config(event=event, lens=lens, source=source)
    cm = _DummyConfigManager(cfg)
    MulensEvent(event, cm)
    Lens(lens, cm)
    Source(source, cm)


# ---------------------------------------------------------------------------
# Drift guard: the gate's vocabulary IS the schema's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys, component",
    [
        (bodies.EVENT_KEYS, MulensEvent),
        (bodies.LENS_ENTRY_KEYS, Lens),
        (bodies.SOURCE_ENTRY_KEYS, Source),
    ],
)
def test_known_key_sets_match_config_schema(keys, component):
    """
    Given a component's config_schema(),
    When compared with the gate's known-key set in bodies.py,
    Then they are identical up to the universal 'name' key -- so adding a
      schema key without teaching the gate (or vice versa) fails here
      instead of shipping a key the gate refuses (or one the schema never
      documents).
    """
    schema_keys = {entry["key"] for entry in component.config_schema()}
    assert set(keys) == schema_keys | {"name"}
