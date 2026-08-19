"""A configured `model_root:` must reach every consumer (review 1.9.5).

Three places read the model family's directory. Two of them hardcoded the
PACKAGE root and so could not be pointed anywhere: the Zenodo fetch (which
then downloaded 259 MB into a directory nothing would read) and the plot
module import, which went through the fixed dotted namespace
``exozippy.models.<model>.BCs.plot`` and therefore could not resolve a user
directory at all.

The class the plotter uses is also selected by SUBCLASSING now, not by
being the first ``class`` statement in the file.
"""

import textwrap

import pytest

from exozippy.components.sed.plot import Plot
from exozippy.components.sed.sed import (
    DEFAULT_MODEL_ROOT,
    load_model_plot_module,
    plot_class_from,
)

_NEXTGEN_PLOT = DEFAULT_MODEL_ROOT / "NextGen" / "BCs" / "plot.py"


def _write_model_plot(root, model, body):
    path = root / model / "BCs" / "plot.py"
    path.parent.mkdir(parents=True)
    path.write_text(textwrap.dedent(body))
    return path


def test_the_packaged_model_still_imports_as_a_package_module():
    """
    Given the shipped NextGen plot module,
    When it is loaded,
    Then it comes back under its ordinary dotted name, so its classes are
      identical to those a normal import yields.
    """
    # ARRANGE / ACT
    module = load_model_plot_module(_NEXTGEN_PLOT, "NextGen")

    # ASSERT
    from exozippy.models.NextGen.BCs.plot import NextGenPlot

    assert module.__name__ == "exozippy.models.NextGen.BCs.plot"
    assert plot_class_from(module, _NEXTGEN_PLOT) is NextGenPlot


def test_a_model_outside_the_package_can_be_loaded_at_all(tmp_path):
    """
    Given a model family in a user directory (a configured `model_root:`),
    When its plot module is loaded,
    Then the class comes back.

    Regression: importlib.import_module("exozippy.models.<model>.BCs.plot")
    resolves only inside the installed package, so this was unreachable --
    either an ImportError or, for a name that collides with a packaged
    model, silently the WRONG module.
    """
    # ARRANGE
    path = _write_model_plot(
        tmp_path,
        "MyGrid",
        """
        from exozippy.components.sed.plot import Plot


        class MyGridPlot(Plot):
            pass
        """,
    )

    # ACT
    module = load_model_plot_module(path, "MyGrid")
    cls = plot_class_from(module, path)

    # ASSERT
    assert cls.__name__ == "MyGridPlot"
    assert issubclass(cls, Plot)


def test_a_second_load_reuses_the_module(tmp_path):
    """
    Given an out-of-package model plot module already loaded,
    When it is loaded again,
    Then the same module object comes back rather than being re-executed.
    """
    # ARRANGE
    path = _write_model_plot(
        tmp_path,
        "MyGrid",
        """
        from exozippy.components.sed.plot import Plot

        LOADS = []
        LOADS.append(1)


        class MyGridPlot(Plot):
            pass
        """,
    )

    # ACT
    first = load_model_plot_module(path, "MyGrid")
    second = load_model_plot_module(path, "MyGrid")

    # ASSERT
    assert first is second
    assert first.LOADS == [1]


def test_a_helper_class_defined_first_does_not_become_the_plotter(tmp_path):
    """
    Given a model plot module whose first class statement is a helper,
    When the plot class is selected,
    Then the Plot subclass is chosen, not the helper.

    Regression: the class was picked by ast-parsing the source and taking
    the first ClassDef, so a helper above it was silently instantiated as
    the SED figure's plotter.
    """
    # ARRANGE
    path = _write_model_plot(
        tmp_path,
        "Helpery",
        """
        from exozippy.components.sed.plot import Plot


        class _Helper:
            pass


        class HelperyPlot(Plot):
            pass
        """,
    )

    # ACT
    cls = plot_class_from(load_model_plot_module(path, "Helpery"), path)

    # ASSERT
    assert cls.__name__ == "HelperyPlot"


def test_a_module_with_no_plot_subclass_raises(tmp_path):
    """
    Given a model plot module that defines no Plot subclass,
    When the plot class is selected,
    Then it raises rather than picking whatever class is there.
    """
    # ARRANGE
    path = _write_model_plot(
        tmp_path,
        "Empty",
        """
        class NotAPlot:
            pass
        """,
    )

    # ACT / ASSERT
    with pytest.raises(TypeError, match="0 subclass"):
        plot_class_from(load_model_plot_module(path, "Empty"), path)


def test_a_missing_plot_module_names_the_path(tmp_path):
    """
    Given a model_root with no plot.py for the model,
    When it is loaded,
    Then the error names the path it looked in.
    """
    # ARRANGE
    missing = tmp_path / "Nope" / "BCs" / "plot.py"

    # ACT / ASSERT
    with pytest.raises(FileNotFoundError, match="Nope"):
        load_model_plot_module(missing, "Nope")


def test_the_model_data_fetch_uses_the_configured_root(monkeypatch):
    """
    Given an SED whose model_root is not the package root,
    When the raw model spectra are ensured,
    Then the fetch is pointed at that root.

    Regression: _ensure_model_data passed DEFAULT_MODEL_ROOT even though
    ensure_model_data takes a root, so a `model_root:` user downloaded
    259 MB into a directory nothing else reads -- and then downloaded it
    again into the one that is read.
    """
    # ARRANGE
    from exozippy.components.sed import make_bc
    from exozippy.components.sed.sed import SED

    seen = {}

    def fake_ensure(model, model_root):
        seen["model"] = model
        seen["root"] = model_root

    monkeypatch.setattr(make_bc, "ensure_model_data", fake_ensure)

    sed = object.__new__(SED)
    sed.sedmodel = "NextGen"
    sed.model_root = "/somewhere/else"

    # ACT
    sed._ensure_model_data()

    # ASSERT
    assert seen == {"model": "NextGen", "root": "/somewhere/else"}
