"""
Cross-reference tests for the generated LaTeX macro names.

A generated macro name is ``\\<varname><idx><suffix>``, and it is built in
two different modules: ``components/parameter.py`` EMITS the
``\\providecommand`` (``to_latex_def`` / ``to_latex_prior_def`` /
``to_latex_mode_defs``) while ``outputs/latex.py`` REFERS to it from the
deluxetable body it writes (``build_latex_output`` -> ``_value_cells``).
``run.py`` reuses the mode suffix a third time, for the per-mode plot
filenames.

If those spellings ever disagree the document cites an undefined macro --
"Undefined control sequence" at the very end of a long fit -- or, if the
drift happens to land on a name that does exist, silently prints another
parameter's value.  Review item 4.6 flagged exactly that risk: the digit
speller was defined twice (byte-identically, so nothing caught the risk)
and the "modeN" suffix was composed independently in three places.

These tests pin BOTH halves: that there is one implementation of each
naming piece (so the copies cannot drift back apart), and -- the property
with actual teeth -- that every macro the generated table references is
actually defined by the generated variable file, verified statically and,
where a TeX installation exists, by really compiling them.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import exozippy.components.parameter as parameter_module
import exozippy.outputs.latex as latex_module
import exozippy.outputs.modes as modes_module
import exozippy.outputs.texutils as texutils
import exozippy.run as run_module
from exozippy.components.parameter import Parameter, _latex_varname
from exozippy.outputs.latex import build_latex_output
from exozippy.outputs.modes import ModeInfo, ModeReport

# More than nine so the MULTI-digit branch is exercised on both halves of
# the name at once: element 10 -> 'onezero', mode 10 -> 'modeonezero'.
N_MODES = 12
N_ELEMENTS = 12
N_DRAW = 240

# Every macro this project generates carries the 'ez' collision prefix
# (Parameter.latex_prefix) or is one of the mode-weight macros, which use
# the same prefix.  Anything else in the template comes from aastex/LaTeX.
EZ_MACRO = re.compile(r"\\(ez[A-Za-z]*)")
EZ_DEF = re.compile(r"\\providecommand\{\\(ez[A-Za-z]*)\}")


# ----------------------------------------------------------------------
# Fixtures: a component-free system carrying one scalar, one vector and
# one fixed parameter -- enough to exercise every macro-emitting path.
# ----------------------------------------------------------------------


class _FakeComp:
    label = "star"


def _fake_system(comp):
    class _Sys:
        name = "test"

        def get_all_components(self):
            return [comp]

    return _Sys()


def _mode_report(n_modes=N_MODES):
    modes = [
        ModeInfo(
            index=k,
            weight=1.0 / n_modes,
            n_draws=N_DRAW // n_modes,
            lp_med=0.0,
            lp_max=0.0,
            delta_lp_max=0.0,
            per_chain_weight=np.array([1.0 / n_modes]),
            weight_err=0.01,
        )
        for k in range(n_modes)
    ]
    return ModeReport(
        labels=np.zeros((1, N_DRAW), dtype=int),
        modes=modes,
        n_valid=N_DRAW,
        n_invalid=0,
        n_unassigned=0,
        provenance="occupancy (test)",
        weights_reliable=True,
        n_transitions=100,
        feature_vars=["a_raw"],
    )


def _build_system_with_params():
    """A system whose one component carries every macro-emitting shape."""
    rng = np.random.default_rng(0)
    comp = _FakeComp()

    scalar = Parameter(
        label="star.teff",
        latex=r"T_{\rm eff}",
        description="Effective temperature",
        initval=5000.0,
        lower=3000.0,
        upper=7000.0,
    )
    scalar.posterior = 5000.0 + rng.normal(0, 10, N_DRAW)

    vector = Parameter(
        label="rvinstrument.jitter_variance",
        latex=r"\sigma_J^2",
        description="RV jitter variance",
        initval=[1.0] * N_ELEMENTS,
        lower=[0.0] * N_ELEMENTS,
        upper=[10.0] * N_ELEMENTS,
        shape=(N_ELEMENTS,),
        names=[f"INST{i}" for i in range(N_ELEMENTS)],
    )
    vector.posterior = 1.0 + rng.normal(0, 0.1, (N_ELEMENTS, N_DRAW))

    # Fixed (never sampled): keeps its single unsuffixed macro, which the
    # multimodal table spans across every mode column with \multicolumn.
    fixed = Parameter(
        label="star.age",
        latex=r"{\rm Age}",
        description="Stellar age",
        initval=4.6,
        lower=0.0,
        upper=13.8,
    )

    comp.teff = scalar
    comp.jitter_variance = vector
    comp.age = fixed
    system = _fake_system(comp)
    system.mode_labels = np.tile(np.arange(N_MODES), N_DRAW // N_MODES)
    return system


def _generate(tmp_path, mode_report):
    var_path = tmp_path / "vars.tex"
    tmpl_path = tmp_path / "tmpl.tex"
    build_latex_output(
        _build_system_with_params(),
        var_filename=str(var_path),
        template_filename=str(tmpl_path),
        mode_report=mode_report,
    )
    return var_path.read_text(), tmpl_path.read_text()


# ----------------------------------------------------------------------
# One implementation of each naming piece
# ----------------------------------------------------------------------


def test_mode_suffix_has_a_single_implementation():
    """
    Given the emitter (components/parameter.py), the referrer
      (outputs/latex.py), the plot-filename consumer (run.py) and the
      mode module,
    When each module's mode_suffix is looked up,
    Then all four are the SAME function object, from outputs/texutils.py --
      so the "modeN" suffix cannot be composed independently anywhere.
    """
    shared = texutils.mode_suffix

    assert parameter_module.mode_suffix is shared
    assert latex_module.mode_suffix is shared
    assert modes_module.mode_suffix is shared
    assert run_module.mode_suffix is shared


def test_idx_to_words_has_a_single_implementation():
    """
    Given the digit speller, which builds the <idx> half of every macro
      name and used to exist as two byte-identical copies,
    When the modules that name macros are inspected,
    Then they all resolve to the one texutils implementation and neither
      parameter.py nor modes.py keeps a private copy.
    """
    assert parameter_module.idx_to_words is texutils.idx_to_words
    assert not hasattr(modes_module, "_idx_to_words")
    assert not hasattr(parameter_module, "_idx_to_words")
    # The digit table _latex_varname uses is the same object, so the
    # <varname> and <idx> halves of a name cannot disagree either.
    assert parameter_module.DIGIT_WORDS is texutils.DIGIT_WORDS


def test_mode_suffix_spellings_are_pinned():
    """
    Given mode indices spanning the single- and multi-digit branches,
    When mode_suffix renders them,
    Then the spellings are exactly the ones already written into shipped
      filenames and macro names, and stay consistent with idx_to_words'
      0-based-index/1-based-label convention.
    """
    assert texutils.mode_suffix(0) == "modeone"
    assert texutils.mode_suffix(1) == "modetwo"
    assert texutils.mode_suffix(3) == "modefour"
    assert texutils.mode_suffix(8) == "modenine"
    assert texutils.mode_suffix(9) == "modeonezero"
    assert texutils.mode_suffix(110) == "modeoneoneone"

    for k in range(200):
        assert texutils.mode_suffix(k) == "mode" + texutils.idx_to_words(k + 1)
        assert texutils.mode_word(k) == texutils.idx_to_words(k + 1)


def test_latex_varname_spells_digits_the_same_way():
    """
    Given a parameter label carrying an instance index,
    When _latex_varname builds the macro stem,
    Then its digits are spelled with the shared table, matching how
      idx_to_words spells the element index appended right after it.
    """
    assert _latex_varname("star.0.mass") == "ezstarzeromass"
    assert _latex_varname("lens.12.t_E") == "ezlensonetwotE"
    for n in range(10):
        assert _latex_varname(str(n), prefix="") == texutils.idx_to_words(n)


# ----------------------------------------------------------------------
# THE invariant: emitter and referrer agree on every name
# ----------------------------------------------------------------------


def test_every_macro_the_table_references_is_defined(tmp_path):
    """
    Given a multimodal LaTeX report over scalar, vector and fixed
      parameters, with enough modes and elements to reach the multi-digit
      branch of both index spellings,
    When the generated table body and variable file are compared,
    Then every macro the table references is defined by the variable file
      -- the cross-reference invariant that a divergence between the two
      spellings of <idx> or <suffix> would break.
    """
    var_text, tmpl_text = _generate(tmp_path, _mode_report())

    referenced = set(EZ_MACRO.findall(tmpl_text))
    defined = set(EZ_DEF.findall(var_text))

    assert referenced, "the table referenced no generated macros at all"
    # The multi-digit branch really was reached on both halves.
    assert any(name.endswith("modeonezero") for name in referenced)
    assert any("onezero" in name for name in defined)

    assert not (referenced - defined), (
        "table references undefined macros: "
        + ", ".join(sorted(referenced - defined))
    )


def test_mode_weight_macros_are_defined_for_every_mode(tmp_path):
    """
    Given a multimodal report whose weights all carry a 1-sigma,
    When the weight row and the weight macro defs are generated,
    Then both \\ezmodeweight<word> and \\ezmodeweighterr<word> exist for
      every mode -- these use the bare mode WORD rather than the 'modeN'
      suffix, a second naming convention that must also match end to end.
    """
    var_text, tmpl_text = _generate(tmp_path, _mode_report())

    for k in range(N_MODES):
        word = texutils.mode_word(k)
        assert rf"\providecommand{{\ezmodeweight{word}}}" in var_text
        assert rf"\providecommand{{\ezmodeweighterr{word}}}" in var_text
        assert rf"\ezmodeweight{word} " in tmpl_text


@pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="no TeX installation"
)
def test_generated_macros_actually_compile(tmp_path):
    """
    Given the generated variable file and every macro the table cites,
    When a minimal document \\inputs the definitions and typesets each
      cited macro under pdflatex,
    Then it compiles -- proving the names really do resolve, rather than
      merely matching a regex.

    This is the cheap half, and it runs wherever there is any TeX at all:
    the cross-reference lives entirely in the macro names, so a minimal
    article is enough to prove they resolve. The test below compiles the
    real deluxetable against the vendored aastex701 class.
    """
    var_text, tmpl_text = _generate(tmp_path, _mode_report())
    referenced = sorted(set(EZ_MACRO.findall(tmpl_text)))

    (tmp_path / "vars.tex").write_text(var_text)
    body = "\n".join(rf"\noindent\{name}\par" for name in referenced)
    (tmp_path / "doc.tex").write_text(
        "\\documentclass{article}\n"
        "\\input{vars}\n"
        "\\begin{document}\n" + body + "\n\\end{document}\n"
    )

    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "doc.tex",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    errors = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("!") or "Undefined control sequence" in line
    ]
    assert result.returncode == 0, "pdflatex failed:\n" + "\n".join(errors)
    assert (tmp_path / "doc.pdf").exists()


AASTEX_CLS = Path(__file__).parent / "fixtures" / "tex" / "aastex701.cls"


@pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="no TeX installation"
)
@pytest.mark.skipif(
    not AASTEX_CLS.exists(), reason="vendored aastex701.cls is missing"
)
def test_generated_template_compiles_as_a_real_aastex_document(tmp_path):
    """
    Given the generated deluxetable template and its variable file,
    When they are compiled as an actual aastex701 document,
    Then pdflatex produces a PDF with no errors.

    This is the check the previous test cannot make. That one typesets the
    cited macros in a minimal `article`, which proves the NAMES resolve; it
    says nothing about whether the table body is valid AASTeX -- a stray
    unescaped `%`, a `&` count that disagrees with the column spec, or a
    \\tablecomments{} that swallows the rest of its line all compile
    perfectly well as a list of \\noindent macros and fail here.

    That is not hypothetical. The template shipped a `\\usepackage{apjfonts}`
    line for years; apjfonts is a legacy AASTeX v5 package that is on
    neither CTAN nor TeX Live, so its absence is a fatal "Emergency stop"
    and the generated template could not compile at all except on a machine
    that happened to carry the file. Removing that line is what makes this
    test possible, and this test is what stops it coming back.

    aastex701.cls is vendored under tests/fixtures/tex/ (LPPL 1.3c, ~375 kB)
    so this runs out of the box rather than asking every developer to
    install AASTeX. TEXINPUTS points there; nothing is installed.
    """
    _, tmpl_text = _generate(tmp_path, _mode_report())
    shutil.copy(AASTEX_CLS, tmp_path / AASTEX_CLS.name)
    # _generate writes tmpl.tex next to vars.tex, and the template \inputs
    # the variable file by stem -- so both must be compiled in place.

    assert r"\documentclass{aastex701}" in tmpl_text
    assert "apjfonts" not in tmpl_text, (
        "the template loads apjfonts again -- it is not on CTAN or in TeX "
        "Live, so this makes the generated file uncompilable for anyone "
        "who does not already have that file (see build_latex_output)"
    )

    env = dict(os.environ, TEXINPUTS=f".:{tmp_path}:")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "tmpl"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    errors = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("!") or "Undefined control sequence" in line
    ]
    assert result.returncode == 0, "pdflatex failed:\n" + "\n".join(errors)
    assert (tmp_path / "tmpl.pdf").exists()
