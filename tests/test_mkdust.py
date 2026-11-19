"""exozippy-mkdust: the dust-map prior tool (utilities/mkdust.py).

No test touches the network: the map files are tiny fixtures written here in
the maps' own formats, and the VizieR/IRSA queries are monkeypatched.
"""

import re

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from exozippy.outputs.texutils import known_bib_keys
from exozippy.utilities import mkdust as md

# KMT-2024-BLG-2087: (l, b) = (0.6374, -1.7011).
RA, DEC = "17:53:48.01", "-29:15:47.48"


def _coord():
    return md.parse_coord(RA, DEC)


def _lb():
    return md._lb(_coord())


def _zel_csv(tmp_path, flag=0, dl_arcmin=0.5):
    """A three-pixel Zelakiewicz+2025 map: one near the target, two far."""
    l0, b0 = _lb()
    rows = [
        (l0 + dl_arcmin / 60.0, b0, 0.216, 0.050, 0.048, 0.20, 0.04, flag),
        (l0 + 0.5, b0, 0.40, 0.05, 0.05, 0.3, 0.04, 0),
        (l0, b0 + 0.5, 0.10, 0.05, 0.05, 0.1, 0.04, 0),
    ]
    path = tmp_path / "ukirt_map.csv"
    with open(path, "w") as f:
        f.write("l,b,aks,aks_err_up,aks_err_low,ehks,ehks_err,Flag\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return path


def _nataf_table3(tmp_path, s_evi=0.157):
    """A Nataf+2013 Table 3 with an UNRELIABLE point nearest the target."""
    c = _coord()
    ra_h = c.ra.deg / 15.0
    l0, b0 = _lb()
    rows = [
        # QualityFlag 1, 5" away: must be ignored.
        (
            l0,
            b0,
            ra_h + 5.0 / 3600 / 15,
            c.dec.deg,
            9.9,
            9.9,
            0.1,
            14.5,
            0.3,
            0.34,
            1,
        ),
        # QualityFlag 0, 62" away.
        (
            l0,
            b0,
            ra_h,
            c.dec.deg + 62.0 / 3600,
            1.847,
            1.516,
            s_evi,
            14.5,
            0.3,
            0.344,
            0,
        ),
    ]
    path = tmp_path / "Table3.dat"
    with open(path, "w") as f:
        f.write(
            "### lgalac bgalac RA DEC A_{I} E(V-I) S_{E(V-I)} mu S_{mu} "
            "RJKVI Quality\n"
        )
        for r in rows:
            f.write(" ".join(repr(float(x)) for x in r) + "\n")
    return path


def _surot_table(rows):
    return Table(
        rows=rows,
        names=("E(J-Ks)", "e_E(J-Ks)", "_r"),
        dtype=(float, float, float),
    )


# --- the conversions ----------------------------------------------------------


def test_the_nataf_ratios_give_the_documented_av_per_ejk():
    """
    Given Nataf+2013's bulge means A_I/E(V-I) = 1.217, R_JKVI = 0.3433,
    When A_V/E(J-Ks) and A_V/A_Ks are formed,
    Then they are the 6.458 and 15.30 the output notes and the doc quote.
    """
    assert md.AV_PER_EJK == pytest.approx(6.458, abs=5e-4)
    assert md.AV_PER_EJK / md.SUROT20_AKS_PER_EJK == pytest.approx(
        15.30, abs=5e-3
    )


# --- each reader --------------------------------------------------------------


def test_surot_takes_the_nearest_node_and_converts_through_nataf(monkeypatch):
    """
    Given a VizieR cone with two Surot nodes,
    When read_surot2020 reads it,
    Then the NEAREST node is used; A_V = AV_PER_EJK E(J-Ks) with the map
      error and the 7% ratio scatter in quadrature; the Ks line is Surot's
      own vector in VVV Ks.
    """
    table = _surot_table([(2.0, 0.5, 0.9), (0.5, 0.01, 0.1)])
    monkeypatch.setattr(md, "_vizier_cone", lambda *a: table)

    m = md.read_surot2020(_coord())

    assert m.status == "ok"
    assert m.av.mu == pytest.approx(md.AV_PER_EJK * 0.5)
    assert m.av.sigma == pytest.approx(
        m.av.mu * np.hypot(0.01 / 0.5, md.AV_PER_EJK_FRAC_SIGMA)
    )
    assert m.av.band is None
    (ks,) = m.native
    assert ks.band == md.VVV_KS
    assert ks.mu == pytest.approx(md.SUROT20_AKS_PER_EJK * 0.5)


def test_surot_outside_and_a_changed_format(monkeypatch):
    """
    Given no node in the cone, then a table missing the E(J-Ks) column,
    When read_surot2020 reads each,
    Then the first is "outside" and the second raises MapFormatError naming
      the column (never a guessed column).
    """
    monkeypatch.setattr(md, "_vizier_cone", lambda *a: None)
    assert md.read_surot2020(_coord()).status == "outside"

    bad = Table(rows=[(1.0, 0.1)], names=("EJK", "_r"))
    monkeypatch.setattr(md, "_vizier_cone", lambda *a: bad)
    with pytest.raises(md.MapFormatError, match=r"E\(J-Ks\)"):
        md.read_surot2020(_coord())


def test_zelakiewicz_reads_the_pixel_holding_the_target(tmp_path):
    """
    Given a three-pixel map with one pixel 30" from the target,
    When read_zelakiewicz2025 reads it,
    Then that pixel's A_Ks is used, A_V goes through Surot's vector and the
      Nataf ratio, and its width carries the extra A_Ks -> E(J-Ks) term.
    """
    m = md.read_zelakiewicz2025(_coord(), _zel_csv(tmp_path))

    assert m.status == "ok"
    (ks,) = m.native
    assert ks.mu == pytest.approx(0.216) and ks.sigma == pytest.approx(0.049)
    assert m.av.mu == pytest.approx(
        0.216 / md.SUROT20_AKS_PER_EJK * md.AV_PER_EJK
    )
    frac = np.sqrt(
        (0.049 / 0.216) ** 2
        + md.AKS_TO_EJK_FRAC_SIGMA**2
        + md.AV_PER_EJK_FRAC_SIGMA**2
    )
    assert m.av.sigma == pytest.approx(m.av.mu * frac)


def test_zelakiewicz_outside_and_flagged(tmp_path):
    """
    Given the nearest pixel 3' away, then a flagged (lower-limit) pixel,
    When read,
    Then the first is "outside"; the second is "lower_limit", says so, and
      is never chosen as the default.
    """
    far = md.read_zelakiewicz2025(_coord(), _zel_csv(tmp_path, dl_arcmin=3))
    assert far.status == "outside" and far.av is None

    flagged = md.read_zelakiewicz2025(_coord(), _zel_csv(tmp_path, flag=1))
    assert flagged.status == "lower_limit"
    assert "LOWER LIMIT" in flagged.detail
    assert md.choose_default([flagged], _lb()[1]) is None


def test_nataf2013_skips_unreliable_points_and_sums_ai_and_evi(tmp_path):
    """
    Given Table 3 with an unreliable point nearest and a reliable one 62"
      away,
    When read_nataf2013 reads it,
    Then the reliable point is used: A_V = A_I + E(V-I), its width from the
      differential reddening, and the A_I line in Cousins I.
    """
    m = md.read_nataf2013(_coord(), _nataf_table3(tmp_path))

    assert m.status == "ok"
    assert m.av.mu == pytest.approx(1.847 + 1.516)
    assert m.av.sigma == pytest.approx((1.847 / 1.516 + 1.0) * 0.157)
    (ai,) = m.native
    assert ai.band == md.COUSINS_I and ai.mu == pytest.approx(1.847)
    assert '62"' in m.detail


def test_nataf2013_with_no_differential_reddening_uses_the_granularity(
    tmp_path,
):
    """A sight line listing S_E(V-I) = 0 gets Nataf's 9%, and says so."""
    m = md.read_nataf2013(_coord(), _nataf_table3(tmp_path, s_evi=0.0))
    assert m.av.sigma == pytest.approx(0.09 * (1.847 + 1.516))
    assert "9%" in m.av.note


# --- choosing and writing -----------------------------------------------------


def _readings(tmp_path, monkeypatch, surot_ok=True):
    if surot_ok:
        table = _surot_table([(0.5, 0.01, 0.1)])
        monkeypatch.setattr(md, "_vizier_cone", lambda *a: table)
    else:

        def _down(*a):
            raise OSError("404 Client Error")

        monkeypatch.setattr(md, "_vizier_cone", _down)
    zel = _zel_csv(tmp_path)
    nat = _nataf_table3(tmp_path)
    monkeypatch.setattr(md, "zelakiewicz_path", lambda: zel)
    monkeypatch.setattr(md, "nataf2013_path", lambda: nat)

    def _sfd(coord):
        m = md.MapReading("sfd", "SFD", "ok")
        m.av = md.Prior(3.0, 0.1, ("Schlafly:2011", "Schlegel:1998"))
        return m

    monkeypatch.setattr(md, "read_sfd", _sfd)
    return md.query_maps(_coord())


def _live_and_commented(text):
    live = yaml.safe_load(text)
    commented = [
        yaml.safe_load(line[2:])
        for line in text.splitlines()
        if line.startswith("# sed.dust_anchor:")
    ]
    return live, commented


def test_the_file_has_one_live_prior_and_every_map_commented(
    tmp_path, monkeypatch
):
    """
    Given every map answering,
    When the file is written,
    Then it parses as YAML with exactly ONE live key, sed.dust_anchor, from
      Surot (the default) as A_V; every other candidate is a commented line
      that parses once uncommented; every citation is a references.bib key;
      and the maps it cannot read are listed as not queried.
    """
    readings = _readings(tmp_path, monkeypatch)
    out = md.mkdust(RA, DEC, name="t", outpath=tmp_path, readings=readings)
    text = out.read_text()

    live, commented = _live_and_commented(text)
    assert list(live) == ["sed.dust_anchor"]
    assert live["sed.dust_anchor"]["mu"] == pytest.approx(
        md.AV_PER_EJK * 0.5, abs=1e-3
    )
    assert live["sed.dust_anchor"]["citation"] == ["Surot:2020", "Nataf:2013"]
    # Surot Ks; Zelakiewicz A_V + Ks; Nataf A_V + A_I; SFD A_V.
    assert len(commented) == 6
    bib = known_bib_keys()
    for entry in [
        live["sed.dust_anchor"],
        *[c["sed.dust_anchor"] for c in commented],
    ]:
        assert set(entry) == {"mu", "sigma", "citation"}
        assert entry["sigma"] > 0
        assert set(entry["citation"]) <= bib
    assert "[band Paranal/VISTA.Ks]" in text
    assert "dust_anchor_band" in text
    for title, _ in md.SKIPPED:
        assert title in text


def test_a_failed_query_is_reported_and_the_next_map_is_the_default(
    tmp_path, monkeypatch, caplog
):
    """
    Given VizieR down,
    When the maps are queried and written,
    Then Surot is listed FAILED with the error, a WARNING says so, and the
      live line is Zelakiewicz's A_V with the header naming the preferred
      map it passed over.
    """
    readings = _readings(tmp_path, monkeypatch, surot_ok=False)
    text = md.render(_coord(), readings, "t")

    assert readings[0].status == "failed"
    assert "404" in readings[0].detail
    assert any(
        r.levelname == "WARNING" and "Surot" in r.message
        for r in caplog.records
    )
    live, _ = _live_and_commented(text)
    assert "Zelakiewicz:2025" in live["sed.dust_anchor"]["citation"]
    assert re.search(r"Preferred but unusable here.*Surot", text)


def test_sfd_is_the_default_only_off_the_bulge():
    """
    Given only an SFD reading,
    When the default is chosen at |b| = 1.7 and at |b| = 20,
    Then there is no default inside |b| < 5 (SFD is not valid there) and SFD
      is the default outside.
    """
    m = md.MapReading("sfd", "SFD", "ok")
    m.av = md.Prior(3.0, 0.1, ("Schlafly:2011",))
    assert md.choose_default([m], -1.7) is None
    assert md.choose_default([m], -20.0) is m


def test_the_cli_takes_a_negative_sexagesimal_dec(tmp_path, monkeypatch):
    """
    Given `exozippy-mkdust 17:53:48.01 -29:15:47.48`,
    When main parses it,
    Then argparse does not take the Dec for an option, and the written
      position is the target's.
    """
    seen = {}

    def _query(coord):
        seen["coord"] = coord
        return []

    monkeypatch.setattr(md, "query_maps", _query)
    md.main([RA, DEC, "--name", "cli", "--outpath", str(tmp_path)])

    target = SkyCoord(RA, DEC, unit=(u.hourangle, u.deg))
    assert seen["coord"].separation(target).arcsec < 1e-3
    text = (tmp_path / "cli.dust.params.yaml").read_text()
    assert "No map gave a usable default" in text
    assert yaml.safe_load(text) is None


def test_the_live_line_is_a_prior_the_dust_profile_reads(
    tmp_path, monkeypatch
):
    """
    Given the tool's live line pasted into a lens + source + clump params
      file (the dust profile on),
    When the system is prepared,
    Then sed.dust_anchor resolves to that mu and sigma -- the line feeds the
      existing channel, no new key.
    """
    from test_dust_profile import _lens_params, _system

    readings = _readings(tmp_path, monkeypatch)
    live, _ = _live_and_commented(md.render(_coord(), readings, "t"))
    system = _system(tmp_path, params=_lens_params(**live))

    assert system.sed._dust_plan.on
    resolved = system.config_manager.resolve("sed", "dust_anchor")
    assert resolved["mu"] == pytest.approx(live["sed.dust_anchor"]["mu"])
    assert resolved["sigma"] == pytest.approx(live["sed.dust_anchor"]["sigma"])
