"""The NextGen pipeline's step 0 (fetch_raw_spectra: SVO's index is the
authority on which nodes exist) and step 1b (pad_missing_nodes: the grid
nodes no spectrum exists for take the nearest lower-gravity plane's BC
rows).  No network: the index is a VOTable string, the tables synthetic.
"""

import numpy as np
import pandas as pd
import pytest

from exozippy.components.sed.bc_grid import (
    bc_table_path,
    complete_axes,
    read_bc_meta,
    read_bc_table,
    write_bc_table,
)
from exozippy.models.NextGen import fetch_raw_spectra as fetch
from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen

_ROW = (
    "<TR>\n<TD>{teff}</TD><TD>{logg}</TD><TD>{feh}</TD><TD>{alpha}</TD>"
    "<TD>51890</TD><TD>desc</TD><TD>1</TD><TD>2</TD><TD>3</TD><TD>4</TD>"
    "<TD><![CDATA[https://svo2.cab.inta-csic.es/theory/newov2/ssap.php?"
    "model=bt-nextgen-agss2009&fid={fid}]]></TD><TD>x</TD><TD>1</TD></TR>\n"
)


def _votable(rows):
    body = "".join(_ROW.format(**r) for r in rows)
    return f"<VOTABLE><RESOURCE><TABLE><DATA><TABLEDATA>{body}</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"


def test_the_ssap_index_parses_to_nodes_and_the_alpha_fallback_is_honored():
    """
    Given SVO's SSAP index as it is served (positional TDs, the file id in
      the access URL, alpha blank on the plain grid),
    When it is parsed,
    Then every spectrum row becomes (teff, logg, feh, alpha, fid), the
      available nodes are what the index lists, and a node with only an
      alpha-enhanced spectrum resolves to it in ALPHA_GRID_PTS order.
    """
    text = _votable(
        [
            dict(teff=2600, logg=5.5, feh=0, alpha="", fid=196),
            dict(teff=2600, logg=5.5, feh=-0.5, alpha=0.2, fid=197),
            dict(teff=2600, logg=5.5, feh=-0.5, alpha=0.4, fid=198),
        ]
    )

    index = fetch.parse_ssap_index(text)

    assert list(index.columns) == ["teff", "logg", "feh", "alpha", "fid"]
    assert len(index) == 3
    assert fetch.available_nodes(index) == {
        (2600.0, 5.5, 0.0),
        (2600.0, 5.5, -0.5),
    }
    assert fetch.fid_for_node(index, 2600, 5.5, 0.0) == (0.0, 196)
    assert fetch.fid_for_node(index, 2600, 5.5, -0.5) == (0.2, 197)
    with pytest.raises(KeyError, match="no BT-NextGen spectrum"):
        fetch.fid_for_node(index, 2700, 5.5, 0.0)
    assert fetch.spectrum_url(196).endswith("&fid=196&format=ascii")


def test_an_index_with_no_rows_raises():
    with pytest.raises(ValueError, match="no spectrum rows"):
        fetch.parse_ssap_index("<VOTABLE></VOTABLE>")


def _synthetic_table(
    root, grid, facility="2MASS", cols=("2MASS_J", "2MASS_H")
):
    """A complete table on grid['logg'][:-1] (the last plane absent), with
    values that encode their node so a copied row is recognizable."""
    rows = []
    for z in grid["feh"]:
        for g in grid["logg"][:-1]:
            for t in grid["teff"]:
                for a in grid["av"]:
                    rows.append(
                        {
                            "teff": t,
                            "logg": g,
                            "feh": z,
                            "alpha": 0.2 if g == 5.0 else 0.0,
                            "Av": a,
                            "Rv": 3.1,
                            **{
                                c: 100 * k + t / 1000 + g + z + a
                                for k, c in enumerate(cols)
                            },
                        }
                    )
    path = bc_table_path(root, "NextGen", facility)
    write_bc_table(
        pd.DataFrame(rows),
        path,
        {
            c: {
                "svo_id": c,
                "generator": gen.GENERATOR,
                "spectra": gen.SPECTRA_TAG,
            }
            for c in cols
        },
        table_meta={"facility": facility, "model": "NextGen"},
        allow_new_nodes=True,
    )
    return path


def test_pad_copies_the_nearest_lower_plane_and_records_it(
    tmp_path, monkeypatch
):
    """
    Given a 2MASS table complete on logg 4.5 and 5.0, a grid with a 5.5
      plane, and an index that has 5.5 spectra only for the cool Teff,
    When the missing nodes are padded,
    Then the hot (teff, 5.5, feh) rows equal the (teff, 5.0, feh) rows at
      every Av and filter, alpha included (the provenance of the spectrum
      used), the cool ones are left for step 2, meta['padding'] names each
      padded node and its source plane, the loader's complete box still
      excludes the half-empty plane until step 2 fills it, and a rerun pads
      nothing more.
    """
    grid = {
        "teff": np.array([4000.0, 5000.0, 6000.0]),
        "logg": np.array([4.5, 5.0, 5.5]),
        "feh": np.array([-0.5, 0.0]),
        "av": np.array([0.0, 1.0]),
    }
    root = tmp_path / "models"
    path = _synthetic_table(root, grid)
    monkeypatch.setattr(
        gen,
        "unavailable_grid_nodes",
        lambda g=None, raw_path=None: [(6000.0, 5.5, z) for z in grid["feh"]],
    )
    monkeypatch.setattr(gen, "ensure_bc_tables", lambda *a, **k: None)
    monkeypatch.setattr(gen, "read_grid_yaml", lambda model_root=None: grid)

    counts = gen.pad_missing_nodes(
        {"2MASS": ["2MASS/2MASS.J", "2MASS/2MASS.H"]}, model_root=root
    )

    assert counts == {"2MASS": 4}  # 2 feh x 2 Av
    df = read_bc_table(path)
    for z in grid["feh"]:
        for a in grid["av"]:
            pad = df[
                (df.teff == 6000)
                & (df.logg == 5.5)
                & (df.feh == z)
                & (df.Av == a)
            ]
            src = df[
                (df.teff == 6000)
                & (df.logg == 5.0)
                & (df.feh == z)
                & (df.Av == a)
            ]
            assert len(pad) == 1 and len(src) == 1
            for c in ("2MASS_J", "2MASS_H", "alpha", "Rv"):
                assert float(pad[c].iloc[0]) == float(src[c].iloc[0])
    assert df[(df.logg == 5.5) & (df.teff < 6000)].empty
    padding = read_bc_meta(path)["padding"]
    assert sorted(
        (p["teff"], p["logg"], p["feh"], p["from_logg"]) for p in padding
    ) == [(6000.0, 5.5, -0.5, 5.0), (6000.0, 5.5, 0.0, 5.0)]
    box = complete_axes(df, ["2MASS_J", "2MASS_H"])
    assert box["logg_pts"].max() == 5.0
    assert gen.pad_missing_nodes(
        {"2MASS": ["2MASS/2MASS.J", "2MASS/2MASS.H"]}, model_root=root
    ) == {"2MASS": 0}
