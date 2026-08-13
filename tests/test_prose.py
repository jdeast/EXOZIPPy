"""Tests for the modeling-prose collector, writer, and citation library.

Three layers, mirroring the feature (see notes/modeling_prose.txt):

* the collector (``outputs/prose.py``): idempotency, ordering, section
  validation, citation extraction;
* THE cross-reference with teeth, in the mold of test_latex_macro_xref:
  every ``\\cite*{key}`` emitted anywhere in the shipped source must have
  an entry in the universal ``references.bib`` -- a sentence citing a
  missing key fails here, not in the user's bibtex run at the end of a
  multi-day fit;
* the writer (``outputs/modeling.py``): regenerate-not-append semantics,
  fragments and figures included exactly when they exist on disk, and --
  where a TeX installation exists -- the generated document really
  compiling, bibliography and all, from the shipped support files.

Plus one integration test on a real example topology (ob08092), pinning
that the sentences a config implies are present and the ones it does not
imply are absent.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from exozippy.outputs import modeling
from exozippy.outputs.prose import (
    POST_FIT_SECTIONS,
    SECTION_ORDER,
    ProseCollector,
    extract_cite_keys,
    join_names,
    plural,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "exozippy"
BIB = SRC / "latex" / "references.bib"

# Any \cite / \citet / \citep / \citealt (+ starred, + [pre][post] args)
# appearing in shipped source, including inside escaped string literals.
CITE_RE = re.compile(r"\\+cite[a-zA-Z]*\*?(?:\[[^\]]*\]){0,2}\{([^{}]+)\}")
BIB_KEY_RE = re.compile(r"@\w+\{([^,\s]+),")


# ----------------------------------------------------------------------
# Collector
# ----------------------------------------------------------------------


def test_add_is_idempotent_and_last_write_wins():
    """
    Given two add() calls sharing a key,
    When the collector is read back,
    Then one sentence exists, carrying the LAST text, in its ORIGINAL
      insertion position -- regeneration must neither duplicate nor
      shuffle the paragraph.
    """
    c = ProseCollector()
    c.add("first version.", section="data", key="k1")
    c.add("second sentence.", section="data", key="k2")
    c.add("updated version.", section="data", key="k1")

    texts = [s.text for s in c.sentences()]
    assert texts == ["updated version.", "second sentence."]
    assert len(c) == 2


def test_unknown_section_raises():
    """
    Given a sentence naming a section outside the vocabulary,
    When add() is called,
    Then it raises -- a silently dropped sentence is a modeling choice the
      draft never mentions, the exact failure mode the feature exists to
      prevent.
    """
    c = ProseCollector()
    with pytest.raises(ValueError, match="Unknown prose section"):
        c.add("orphan.", section="observations")


def test_sentences_sort_by_section_then_rank_then_insertion():
    """
    Given sentences added out of document order,
    When sentences() is read,
    Then they come back grouped by SECTION_ORDER, by rank within a
      section, insertion order breaking ties.
    """
    c = ProseCollector()
    c.add("sampling.", section="sampling")
    c.add("data late.", section="data", rank=90)
    c.add("data lead.", section="data", rank=10)
    c.add("data mid a.", section="data")
    c.add("data mid b.", section="data")

    texts = [s.text for s in c.sentences()]
    assert texts == [
        "data lead.",
        "data mid a.",
        "data mid b.",
        "data late.",
        "sampling.",
    ]
    assert [sec for sec, _ in c.paragraphs()] == ["data", "sampling"]


def test_cite_extraction_handles_natbib_forms():
    """
    Given the natbib forms components actually write,
    When keys are extracted,
    Then all keys appear once each, in first-appearance order.
    """
    text = (
        r"\citet{Chen:2017} and \citep[e.g.][]{Batista:2011, Gould:2004} "
        r"and \citealt*{Chen:2017}"
    )
    assert extract_cite_keys(text) == [
        "Chen:2017",
        "Batista:2011",
        "Gould:2004",
    ]


def test_join_names_and_plural():
    """Given 0-3 names, When joined, Then Oxford-comma English."""
    assert join_names([]) == ""
    assert join_names(["A"]) == "A"
    assert join_names(["A", "B"]) == "A and B"
    assert join_names(["A", "B", "C"]) == "A, B, and C"
    assert plural(1, "dataset") == "1 dataset"
    assert plural(2, "dataset") == "2 datasets"


def test_software_list_is_idempotent():
    """Given repeated add_software, When read, Then one copy each."""
    c = ProseCollector()
    c.add_software("celerite2")
    c.add_software("celerite2")
    c.add_software("MulensModel")
    assert c.software == ["celerite2", "MulensModel"]


# ----------------------------------------------------------------------
# THE cross-reference: every emitted cite key exists in references.bib
# ----------------------------------------------------------------------


def _bib_keys():
    text = BIB.read_text()
    return [m.group(1) for m in BIB_KEY_RE.finditer(text)]


def test_references_bib_parses_and_has_unique_keys():
    """
    Given the shipped universal references.bib,
    When its entry keys are extracted,
    Then entries exist and no key is defined twice (bibtex would warn and
      pick one arbitrarily).
    """
    keys = _bib_keys()
    assert len(keys) > 20
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"duplicate bibtex keys: {sorted(dupes)}"


def test_every_cite_key_in_source_exists_in_references_bib():
    """
    Given every \\cite*{...} in the shipped source (prose sentences,
      table_notes, PriorContribution strings, docstrings),
    When the keys are collected,
    Then each has an entry in references.bib -- add the entry in the same
      commit as the sentence, or the generated draft cites a reference
      bibtex cannot resolve.
    """
    known = set(_bib_keys())
    # Keys must look like keys: this raw-text scan also sweeps docstrings,
    # where ``\\citet{...}`` placeholders and cite braces broken across
    # string literals produce junk captures.  The runtime half (the
    # ob08092 integration test below) checks the ACTUAL collected prose
    # with no such filter.
    key_re = re.compile(r"^[A-Za-z][A-Za-z0-9:_-]*$")
    missing = {}
    for path in SRC.rglob("*.py"):
        for m in CITE_RE.finditer(path.read_text()):
            for key in m.group(1).split(","):
                key = key.strip()
                if key_re.match(key) and key not in known:
                    missing.setdefault(key, path.name)
    assert not missing, (
        "cite keys with no references.bib entry (add them to "
        "src/exozippy/latex/references.bib): "
        + ", ".join(f"{k} ({f})" for k, f in sorted(missing.items()))
    )


def test_provenance_text_escapes_for_latex_text_mode():
    """
    Given the mode-weight provenance's actual plain-text vocabulary
      (N_eff, >=),
    When it is escaped for a \\tablecomments or a prose sentence,
    Then the underscore is escaped (a bare one is a hard 'Missing $
      inserted') and the comparison operators are wrapped in math mode
      (bare < > silently render as inverted punctuation under OT1).

    Regression: a real multimodal fit's table was uncompilable until the
    modeling draft's auto-compile first exercised it.
    """
    from exozippy.outputs.texutils import latex_escape_prose

    out = latex_escape_prose("N_eff for the weights >= 4.6, q < 1")
    assert out == r"N\_eff for the weights $\geq$ 4.6, q $<$ 1"


def test_descriptions_are_valid_latex():
    """
    Given every ``description:`` in the shipped defaults.yaml files,
    When their text spans (everything outside $...$ math) are scanned,
    Then no LaTeX special appears raw -- descriptions are TRUSTED LaTeX
      now (they may carry math like $\\log_{10}{M_P/M_\\star}$, so they are
      no longer escaped on their way into the table), which makes a raw
      underscore in one a hard 'Missing $ inserted' at the end of a fit.
    """
    import yaml

    bad = []
    for path in SRC.rglob("defaults.yaml"):

        def walk(node, crumbs):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "description" and isinstance(v, str):
                        text = re.sub(r"\$[^$]*\$", "", v)
                        raw = set("_&%#^~{}<>|") & set(
                            text.replace(r"\_", "").replace(r"\%", "")
                        )
                        if raw:
                            bad.append(
                                f"{path.relative_to(SRC)}: "
                                f"{'.'.join(crumbs)}: {v!r} (raw {raw})"
                            )
                    else:
                        walk(v, crumbs + [str(k)])

        walk(yaml.safe_load(path.read_text()), [])
    assert not bad, "descriptions with raw LaTeX specials:\n" + "\n".join(bad)


def test_post_fit_sections_are_a_subset_of_the_vocabulary():
    """The writer's Results routing must stay inside SECTION_ORDER."""
    assert set(POST_FIT_SECTIONS) <= set(SECTION_ORDER)


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------


class _FakeSpec:
    def __init__(self, tag, caption=None, title="A chart"):
        self.id = tag
        self.title = title
        self.meta = {"file_tag": tag}
        if caption is not None:
            self.meta["caption"] = caption


class _FakeComp:
    prefix = "fake"

    def __init__(self, specs):
        self._specs = specs

    def plot_data(self, system, point=None):
        return self._specs


class _FakeSystem:
    name = "FAKE_SYS"

    def __init__(self, comps=()):
        self.prose = ProseCollector()
        self.active_components = {c.prefix: c for c in comps}


def test_build_modeling_output_regenerates_with_what_exists(tmp_path):
    """
    Given one System written at both checkpoints,
    When the table/definitions fragments appear on disk between the two
      calls,
    Then the first document omits the \\input lines and the second carries
      them -- same call, no append, the file is always self-consistent.
    """
    system = _FakeSystem()
    system.prose.add(r"We used \citet{Chen:2017}.", section="planetary")
    prefix = str(tmp_path / "FAKE")

    tex_path = modeling.build_modeling_output(system, prefix)
    first = Path(tex_path).read_text()
    assert "FAKE_definitions" not in first
    assert "FAKE_table" not in first
    assert r"\citet{Chen:2017}" in first
    assert r"\documentclass[twocolumn]{aastex701}" in first
    assert r"\bibliography{references}" in first
    # Support files were copied: the output directory is self-contained.
    for name in modeling.SUPPORT_FILES:
        assert (tmp_path / name).exists()

    (tmp_path / "FAKE_definitions.tex").write_text("% defs\n")
    (tmp_path / "FAKE_table.tex").write_text("% table\n")
    second = Path(modeling.build_modeling_output(system, prefix)).read_text()
    assert r"\input{FAKE_definitions}" in second
    assert r"\input{FAKE_table}" in second


def test_figures_included_exactly_when_their_pdf_exists(tmp_path):
    """
    Given specs with and without rendered posterior PDFs,
    When the document is regenerated,
    Then only the existing PDF gets a figure block, with its declared
      caption (or the generic fallback when the spec has none).
    """
    comp = _FakeComp(
        [
            _FakeSpec("rv", caption=r"The RV curve of \citet{Chen:2017}."),
            _FakeSpec("lc", caption=None, title="Light curve"),
            _FakeSpec("missing", caption="Never rendered."),
        ]
    )
    system = _FakeSystem([comp])
    system.prose.add("Data sentence.", section="data")
    prefix = str(tmp_path / "FAKE")
    Path(prefix + "_mcmc_rv.pdf").write_bytes(b"%PDF-1.4 fake")
    Path(prefix + "_mcmc_lc.pdf").write_bytes(b"%PDF-1.4 fake")

    text = Path(modeling.build_modeling_output(system, prefix)).read_text()

    assert r"\includegraphics[width=\linewidth]{FAKE_mcmc_rv.pdf}" in text
    assert r"The RV curve of \citet{Chen:2017}." in text
    assert r"\includegraphics[width=\linewidth]{FAKE_mcmc_lc.pdf}" in text
    assert "Auto-generated caption" in text  # lc's fallback
    assert "FAKE_mcmc_missing.pdf" not in text


def test_document_sections_appear_in_flow_order(tmp_path):
    """
    Given prose across data, noise, priors and a post-fit section,
    When the document is rendered,
    Then Observations precedes Modeling precedes Results, and post-fit
      sections appear only when populated.
    """
    system = _FakeSystem()
    system.prose.add("DATA-S.", section="data")
    system.prose.add("NOISE-S.", section="noise")
    system.prose.add("PRIOR-S.", section="priors")
    text = modeling.render_document(system, "doc.tex")
    assert r"\section{Results}" not in text
    i_obs = text.index(r"\section{Observations}")
    i_mod = text.index(r"\section{Modeling}")
    assert i_obs < i_mod < text.index("NOISE-S.")
    assert text.index("NOISE-S.") < text.index("PRIOR-S.")

    system.prose.add("CONV-S.", section="convergence")
    text = modeling.render_document(system, "doc.tex")
    assert text.index(r"\section{Results}") > i_mod
    assert "CONV-S." in text


def test_software_line_carries_core_plus_declared(tmp_path):
    """Given add_software declarations, Then \\software has core + extras."""
    system = _FakeSystem()
    system.prose.add("x.", section="data")
    system.prose.add_software("celerite2")
    text = modeling.render_document(system, "doc.tex")
    line = next(l for l in text.splitlines() if l.startswith(r"\software"))
    for name in modeling.CORE_SOFTWARE:
        assert name in line
    assert "celerite2" in line


# ----------------------------------------------------------------------
# End to end: the generated document compiles, bibliography and all
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("bibtex") is None,
    reason="no TeX installation",
)
def test_modeling_document_compiles_with_bibliography(tmp_path):
    """
    Given a document citing real references.bib entries, plus a figure
      and a (fragment) table,
    When compiled with the full pdflatex/bibtex cycle from ONLY the files
      build_modeling_output put in the directory,
    Then a PDF appears and the .bbl resolved the citations -- proving the
      output directory really is self-contained.
    """
    comp = _FakeComp([_FakeSpec("rv", caption="A caption.")])
    system = _FakeSystem([comp])
    system.prose.add(
        r"We imposed the \citet{Chen:2017} relation and sampled with "
        r"PTDE \citep{terBraak:2006, Vousden:2016}.",
        section="planetary",
    )
    prefix = str(tmp_path / "E2E")
    # A real (tiny) PDF for the figure: generated by pdflatex itself so
    # \includegraphics can embed it.
    (tmp_path / "fig.tex").write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}\n"
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "fig.tex"],
        cwd=tmp_path,
        capture_output=True,
        timeout=120,
    )
    os.rename(tmp_path / "fig.pdf", tmp_path / "E2E_mcmc_rv.pdf")
    (tmp_path / "E2E_definitions.tex").write_text("% no macros needed\n")
    (tmp_path / "E2E_table.tex").write_text(
        "\\startlongtable\n"
        "\\begin{deluxetable*}{lc}\n"
        "\\tablehead{\\colhead{a} & \\colhead{b}}\n"
        "\\startdata\n1 & 2 \\\\\n\\enddata\n"
        "\\end{deluxetable*}\n"
    )

    tex_path = modeling.build_modeling_output(system, prefix)
    pdf = modeling.compile_modeling_pdf(tex_path)

    assert pdf is not None and Path(pdf).exists()
    bbl = tmp_path / "E2E_paper.bbl"
    assert bbl.exists()
    for key_fragment in ("Chen", "Braak", "Vousden"):
        assert key_fragment in bbl.read_text()


def test_compile_returns_none_without_pdflatex(tmp_path, monkeypatch):
    """
    Given a machine with no TeX (simulated),
    When compile_modeling_pdf runs,
    Then it returns None without raising -- the .tex sources are the
      deliverable and the PDF is a bonus.
    """
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / "doc.tex").write_text("x")
    assert modeling.compile_modeling_pdf(tmp_path / "doc.tex") is None


# ----------------------------------------------------------------------
# Integration: a real example topology implies exactly these sentences
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def ob08092_system():
    exdir = ROOT / "examples" / "ob08092"
    if not (exdir / "ob08092.yaml").exists():
        pytest.skip("ob08092 example not present")
    from exozippy.system import System

    cfg = yaml.safe_load((exdir / "ob08092.yaml").read_text())
    user_params = yaml.safe_load((exdir / "ob08092.params.yaml").read_text())
    cwd = os.getcwd()
    try:
        os.chdir(exdir)  # config data paths are relative
        system = System(cfg, user_params)
        system.prepare()
        system.build_model()
    finally:
        os.chdir(cwd)
    return system


def test_ob08092_prose_matches_its_topology(ob08092_system):
    """
    Given the built ob08092 system (PSPL microlensing + galacticmodel,
      no transit/GP/robust/Chen),
    When the collector is read,
    Then the sentences its config implies are present, the ones it does
      not are absent, and every cited key resolves in references.bib.
    """
    prose = ob08092_system.prose
    keys = set(prose._sentences)

    # Present: data inventory, err_scale noise, PSPL magnification,
    # flux-space likelihood, IMF + kinematic + event-rate priors.
    assert "mulensinstrument.data" in keys
    assert "mulensinstrument.noise_model" in keys
    assert "mulensinstrument.magnification" in keys
    assert "mulensinstrument.flux_likelihood" in keys
    assert "galacticmodel.imf" in keys
    assert "galacticmodel.kinematic" in keys
    assert "lens.event_rate" in keys

    # PSPL, symbolic path: Paczynski, no VBM/MulensModel software entry.
    mag = prose._sentences["mulensinstrument.magnification"].text
    assert "Paczynski:1986" in mag
    assert "VBMicrolensing" not in prose.software

    # Absent: nothing in this topology asked for them.  The volume prior
    # stands down because galacticmodel covers distance.
    assert "planet.chen" not in keys
    assert "star.volume_prior" not in keys
    assert not any(".gp." in k for k in keys)
    assert not any(".robust." in k for k in keys)

    # Every key cited by the collected prose resolves in the shipped bib.
    known = set(_bib_keys())
    cited = set(prose.cite_keys())
    assert cited, "the topology's prose cited nothing at all"
    assert cited <= known, f"unresolved cite keys: {sorted(cited - known)}"


def test_ob08092_prose_is_idempotent_across_rebuilds(ob08092_system):
    """
    Given the GUI's pattern of a second build_model() on one System,
    When the model is rebuilt,
    Then the collector holds exactly the same sentences (no duplicates,
      no shuffling) -- add() keys make regeneration safe.
    """
    before = [(s.key, s.text) for s in ob08092_system.prose.sentences()]
    ob08092_system.build_model()
    after = [(s.key, s.text) for s in ob08092_system.prose.sentences()]
    assert before == after
