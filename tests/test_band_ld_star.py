"""Which star's limb darkening a band instance carries (review 1.5.3).

Limb darkening is physically a property of a (star, band) pair, but the
coefficients live on the band instance alone, so two hosts sharing one band
instance silently share their limb darkening.  The LOCKED design
(notes/ld_atm_prior.txt) keeps the parameters per band INSTANCE and makes
the pairing explicit instead: every LD consumer registers the star it reads
the limb darkening of, `star_ndx:` on the band block stays the single
source of truth, and it is now validated against (or derived from) those
registrations.

This is the prerequisite PR for the limb-darkening atmosphere prior
(8.5.2), which has to know whose atmosphere a band's coefficients predict.
"""

import numpy as np
import pytest

from exozippy.components.band.band import Band, LDConsumer


# --------------------------------------------------------------------------
# Stub topologies: the resolution rules, without paying for a full build.
# --------------------------------------------------------------------------
class _StubComp:
    def __init__(self, config, **attrs):
        self.config = config
        for k, v in attrs.items():
            setattr(self, k, v)


class _StubSystem:
    def __init__(self, **comps):
        for name, comp in comps.items():
            setattr(self, name, comp)


def _band_for(names, star_ndx=None):
    from conftest import _DummyConfigManager

    cfg = []
    for i, n in enumerate(names):
        entry = {"name": n, "filter": n}
        if star_ndx is not None and star_ndx[i] is not None:
            entry["star_ndx"] = star_ndx[i]
        cfg.append(entry)
    band = Band(cfg, _DummyConfigManager())
    band.load_data(system=None)
    return band


def _stars(*names):
    return _StubComp([{"name": n} for n in names], names=list(names))


def _finite_source_system(source_idx=1, band="I"):
    return _StubSystem(
        star=_stars("Lens", "Source"),
        lens=_StubComp(
            [{"finite_source": True}], source_map=np.array([source_idx])
        ),
        mulensinstrument=_StubComp([{"name": "OGLE", "band": band}]),
    )


def test_mulens_band_resolves_to_the_source_star():
    """
    Given a finite-source microlensing fit whose band declares no star_ndx,
    When the band's LD star is resolved,
    Then it is the SOURCE star -- the surface the magnification actually
    resolves -- not the historical default of 0, which is the LENS.
    """
    band = _band_for(["I"])

    band._resolve_ld_stars(_finite_source_system(source_idx=1))

    assert band.star_indices == [1]
    assert list(band.star_map) == [1]


def test_declared_star_ndx_is_kept_when_it_agrees():
    """
    Given a band that declares the same star its consumer reads,
    When the LD star is resolved,
    Then the declaration stands (star_ndx is still the single source of
    truth; the consumers only validate it).
    """
    band = _band_for(["I"], star_ndx=[1])

    band._resolve_ld_stars(_finite_source_system(source_idx=1))

    assert band.star_indices == [1]


def test_declared_star_ndx_contradicting_a_consumer_raises():
    """
    Given a band declaring star_ndx: 0 while its finite-source microlensing
    consumer reads the limb darkening of source star 1,
    When the LD star is resolved,
    Then it RAISES, naming the band, both stars and the consumer.

    Silently keeping either one is the bug: the coefficients would describe
    one star's atmosphere while the model applied them to another's.
    """
    band = _band_for(["I"], star_ndx=[0])

    with pytest.raises(ValueError) as exc:
        band._resolve_ld_stars(_finite_source_system(source_idx=1))

    msg = str(exc.value)
    assert "'I'" in msg
    assert "star_ndx: 0" in msg
    assert "mulensinstrument[OGLE]" in msg
    assert "Source" in msg


def test_two_consumers_wanting_different_stars_raises(monkeypatch):
    """
    Given one band instance read by two consumers that need different
    stars,
    When the LD star is resolved,
    Then it RAISES and tells the user to define a second band block with
    the same filter -- which is the fix, because the parameters are per
    band INSTANCE and named blocks referencing one filter are already legal.
    """
    band = _band_for(["I"])
    monkeypatch.setattr(
        band,
        "ld_consumers",
        lambda system: [
            LDConsumer("transit[A]", 0, 0),
            LDConsumer("mulensinstrument[OGLE]", 0, 1),
        ],
    )

    with pytest.raises(ValueError) as exc:
        band._resolve_ld_stars(_stub_two_stars())

    msg = str(exc.value)
    assert "more than one star" in msg
    assert "transit[A]" in msg and "mulensinstrument[OGLE]" in msg
    assert "filter: I" in msg


def _stub_two_stars():
    return _StubSystem(star=_stars("A", "B"))


def test_no_consumer_leaves_the_declared_or_default_star():
    """
    Given a band nothing reads the limb darkening of,
    When the LD star is resolved,
    Then the declaration stands, and an undeclared band keeps 0 -- a
    filter-identity-only band is an ordinary configuration and must not
    acquire an opinion it has no basis for.
    """
    band = _band_for(["I", "V"], star_ndx=[2, None])

    band._resolve_ld_stars(_StubSystem(star=_stars("A", "B", "C")))

    assert band.star_indices == [2, 0]


def test_transit_registers_the_planet_host_star():
    """
    Given a transit whose planets all orbit star 1,
    When the LD star is resolved,
    Then the band carries star 1 -- the transited host, which is also the
    star transit._build_dilution deblends against.
    """
    band = _band_for(["TESS"])
    system = _StubSystem(
        star=_stars("A", "B"),
        planet=_StubComp([{"name": "b", "star_ndx": 1}]),
        transit=_StubComp([{"name": "S48", "band": "TESS"}]),
    )

    band._resolve_ld_stars(system)

    assert band.star_indices == [1]


def test_transit_over_several_hosts_warns_rather_than_raising(caplog):
    """
    Given planets around two different stars (so one light curve models
    both) and no star_ndx on the band,
    When the LD star is resolved,
    Then it WARNS and falls back to the historical default rather than
    raising: a light curve models every planet, so its limb darkening is
    ambiguous no matter how many band blocks exist, and "define a second
    band" would be advice the user cannot act on.
    """
    band = _band_for(["TESS"])
    system = _StubSystem(
        star=_stars("A", "B"),
        planet=_StubComp(
            [{"name": "b", "star_ndx": 0}, {"name": "c", "star_ndx": 1}]
        ),
        transit=_StubComp([{"name": "S48", "band": "TESS"}]),
    )

    with caplog.at_level("WARNING"):
        band._resolve_ld_stars(system)

    assert band.star_indices == [0]
    assert "no consumer could name the star" in caplog.text
    assert "star_ndx" in caplog.text


def test_rm_registers_the_orbits_primary_star():
    """
    Given an rvinstrument rm: request on an orbit whose primary is star 1,
    When the LD star is resolved,
    Then the RM band carries star 1 -- the transited star whose line
    profile the Rossiter-McLaughlin distortion comes from.
    """
    band = _band_for(["V"])
    system = _StubSystem(
        star=_stars("A", "B"),
        orbit=_StubComp(
            [{"name": "b"}],
            names=["b"],
            primary_bodies=[[("star", 1)]],
        ),
        rvinstrument=_StubComp([{"name": "TRES", "rm": "b"}]),
    )

    band._resolve_ld_stars(system)

    assert band.star_indices == [1]


# --------------------------------------------------------------------------
# One predicate, two questions.
# --------------------------------------------------------------------------
def test_the_pin_predicate_reads_the_same_consumer_list():
    """
    Given the LD consumer registrations,
    When the unread-LD autopin asks which bands are consumed,
    Then it is exactly the band set of those registrations -- one
    predicate, so a new consumer cannot be remembered for the star and
    forgotten for the pin (or the reverse).
    """
    band = _band_for(["I", "V"])
    system = _finite_source_system(source_idx=1, band="I")

    consumers = band.ld_consumers(system)

    assert {c.band for c in consumers} == band._ld_consumer_indices(system)
    assert band._ld_consumer_indices(system) == {0}
