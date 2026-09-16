"""The one chart style table, and the two copies it must keep honest.

There are two renderers for the same Chart -- ``plotrender.py`` (matplotlib,
the saved PDFs) and ``gui/frontend/src/plotly-adapter.ts`` (plotly, the
browser). Before review 4.11.4 each carried its own palette and role colors
and they agreed *by convention only*. That convention had already broken:
residuals were ``"0.5"`` (#808080) in the PDF and #6e7781 in the GUI.

These tests exist so the agreement is a fact rather than a coincidence:

* the matplotlib shorthands the table replaced resolve to exactly its values,
  so adopting it did not move a single published figure;
* the TypeScript copy is parsed out of the adapter and compared to the Python
  source, so it cannot drift again.

Tests follow AAA with Given/When/Then docstrings.
"""

import json
import re
from pathlib import Path

import pytest
from matplotlib import colors as mcolors
from matplotlib import rcParams

from exozippy import plot_theme

ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "gui"
    / "frontend"
    / "src"
    / "plotly-adapter.ts"
)


def test_the_palette_is_matplotlibs_own_default_cycle():
    """
    Given the shared palette,
    When it is compared to matplotlib's default property cycle,
    Then they are identical.

    This is what makes the change a no-op for every saved PDF: the renderer
    used to hand matplotlib an ``f"C{n}"`` and let its cycle choose, and that
    cycle is tab10. Pinning it here means a future matplotlib that ships a
    different default fails loudly instead of silently re-coloring every
    published figure -- at which point the table stays and the PDFs keep
    their colors, which is the outcome we want.
    """
    # Arrange / Act
    cycle = [
        mcolors.to_hex(c)
        for c in rcParams["axes.prop_cycle"].by_key()["color"]
    ]

    # Assert
    assert list(plot_theme.PALETTE) == cycle


@pytest.mark.parametrize(
    "role,shorthand",
    [
        ("model", "r"),
        ("residual", "0.5"),
    ],
)
def test_role_colors_equal_the_matplotlib_shorthands_they_replaced(
    role, shorthand
):
    """
    Given a role color in the shared table,
    When it is compared to the matplotlib shorthand plotrender used before,
    Then they are the same color.

    The PDFs are what get published, so they are the reference the GUI moved
    onto (JDE, 2026-08-27: "match the pdf"). If either value is ever edited,
    this fails and names which one.
    """
    # Act / Assert
    assert plot_theme.ROLE_COLORS[role] == mcolors.to_hex(shorthand)


def test_the_typescript_copy_has_not_drifted_from_the_python_source():
    """
    Given the adapter's compiled-in copy of the palette and role colors,
    When it is parsed out of the TypeScript,
    Then every value matches the Python table.

    THIS TEST IS THE POINT OF THE ITEM. The defect was not that the two
    tables differed by a shade -- it was that nothing could tell you they
    differed. The copy exists because a first paint must not block on a
    network round-trip (the same reason the CLI must not ask a server what
    color a residual is), so the copy has to be mechanically checked instead.
    """
    # Arrange
    src = ADAPTER.read_text()

    # Act
    palette_block = re.search(r"const TAB10 = \[(.*?)\];", src, re.S)
    assert palette_block, "could not find TAB10 in the adapter"
    palette = re.findall(r'"(#[0-9a-fA-F]{6})"', palette_block.group(1))

    def const(name):
        m = re.search(r'const %s = "(#[0-9a-fA-F]{6})"' % name, src)
        assert m, f"could not find {name} in the adapter"
        return m.group(1)

    # Assert
    assert palette == list(plot_theme.PALETTE)
    assert const("MODEL_COLOR") == plot_theme.ROLE_COLORS["model"]
    assert const("DATA_COLOR") == plot_theme.ROLE_COLORS["data"]
    assert const("RESIDUAL_COLOR") == plot_theme.ROLE_COLORS["residual"]

    alpha = re.search(r"const DATA_ALPHA = ([0-9.]+);", src)
    assert alpha, "could not find DATA_ALPHA in the adapter"
    assert float(alpha.group(1)) == plot_theme.ROLE_ALPHA["data"]


def test_a_user_override_beats_the_palette_and_the_role():
    """
    Given a trace style carrying an explicit color,
    When the style is resolved,
    Then the override wins over both the series index and the role.

    The user's ``plot: {color, marker}`` in the config is the top of the
    precedence order, and it has to survive the table being introduced
    underneath it.
    """
    # Arrange / Act
    forced = plot_theme.resolve(
        "model", series_index=3, overrides={"color": "#123456"}
    )
    indexed = plot_theme.resolve("model", series_index=3)
    bare = plot_theme.resolve("model")

    # Assert
    assert forced["color"] == "#123456"
    assert indexed["color"] == plot_theme.PALETTE[3]
    assert bare["color"] == plot_theme.ROLE_COLORS["model"]


def test_the_palette_wraps_rather_than_raising():
    """
    Given more series than the palette has entries,
    When a high series index is resolved,
    Then it wraps.

    A fit with eleven instruments must not crash on the eleventh, and it must
    wrap the same way in both renderers -- the adapter does
    ``TAB10[i % TAB10.length]``.
    """
    # Act / Assert
    n = len(plot_theme.PALETTE)
    assert (
        plot_theme.resolve("data", series_index=n)["color"]
        == plot_theme.PALETTE[0]
    )
    assert (
        plot_theme.resolve("data", series_index=n + 2)["color"]
        == plot_theme.PALETTE[2]
    )


def test_the_served_copy_is_json_serializable():
    """
    Given the table the GUI serves at /api/theme,
    When it is serialized,
    Then it round-trips as plain JSON.

    The endpoint hands this straight to the browser, so a tuple or a numpy
    scalar sneaking into the table would surface as a 500 rather than as a
    type error here.
    """
    # Act
    payload = json.loads(json.dumps(plot_theme.as_json()))

    # Assert
    assert payload["palette"] == list(plot_theme.PALETTE)
    assert payload["role_colors"] == dict(plot_theme.ROLE_COLORS)
    assert payload["role_alpha"]["data"] == plot_theme.ROLE_ALPHA["data"]
