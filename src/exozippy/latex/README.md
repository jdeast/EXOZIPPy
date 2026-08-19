# LaTeX package data shipped with EXOZIPPy

`aastex701.cls`, `aasjournalv7.bst` and `references.bib` are copied next to every
generated `<prefix>_paper.tex`
by `outputs/modeling.py`, making the output directory a self-contained
LaTeX project: `pdflatex && bibtex && pdflatex && pdflatex` works on any
machine with a bare TeX Live, whether or not AASTeX is installed.  This is
also the ONE vendored copy -- `tests/test_latex_macro_xref.py` and the
modeling compile test point here rather than keeping a fixture copy that
could drift.

## `aastex701.cls` and `aasjournalv7.bst`

AASTeX v7.0.1, 2025-05-09. Copyright 2025 American Astronomical Society,
distributed under the **LaTeX Project Public License 1.3c** (the header of
the `.cls` itself carries the licence), so redistributing them here,
unmodified, is allowed.

- Source: <https://mirrors.ctan.org/macros/latex/contrib/aastex.zip>,
  files `aastex/aastex701.cls` and `aastex/aasjournalv7.bst`
  (retrieved 2026-08-12)
- `aastex701.cls`: 383761 bytes,
  sha256 `c1bb436807f8bb37eb5203056db5a5a60e1c9ae6a3feb6501d7c414b9b08b062`
- `aasjournalv7.bst`: 38788 bytes,
  sha256 `ac5d72dbf3ecb36cd6c651fb514935ccd32f1365bb91085bbfb4de941ebc99a8`

### Why they are in the repo rather than a setup step

The generated `<prefix>_paper.tex` (and the deluxetable fragment it
`\input`s) is a deliverable meant to compile out of the box; a TeX
installation without AASTeX is the common case (this project's own dev box
has `pdflatex` and no `aastex*.cls`).  The same file makes the
compile-the-real-document tests run for anyone who has any TeX at all
instead of only for people who installed AASTeX.  The tests skip when
`pdflatex` is absent, so no TeX installation is required to run the suite
-- which is why CI does not exercise them.

### To update

Refetch from the URL above, replace both files, and update the sizes and
hashes here.  `\documentclass{aastex701}` in `outputs/modeling.py` and
`\bibliographystyle{aasjournalv7}` name the versions, so moving to a later
AASTeX means changing those too.

**Do not add `apjfonts.sty`.** The generated table template used to load
it; it is a legacy AASTeX v5 font package, it is on neither CTAN nor TeX
Live, and its absence is a fatal `Emergency stop` rather than a degraded
render -- so that one line made the generated file uncompilable for anyone
who did not happen to already have it. AASTeX 7 sets its own fonts and
AAS's own `aastex701-sample.tex` does not load it.

## `references.bib`

EXOZIPPy's own universal reference library -- every bibtex entry the
codebase can cite (prose sentences, `table_note`s, `PriorContribution`
strings) plus the software stack.  See its header comment.  It is copied
verbatim, never subset: bibtex selects the cited entries itself, and
uncited entries are a gift to the user.  Add the entry for a new citation
key in the same commit as the sentence that cites it --
`tests/test_prose.py` cross-references every emitted key against this
file.

## `convention.tex`

A drop-in `\section` for the EXOZIPPy microlensing paper, stating the sky
frame, the origins, the parallax signs, the binary geometry (`s`, `q`,
`alpha`) and the mappings onto the other common literature conventions.
Not part of the generated output -- nothing copies it, and `modeling.py`
does not know it exists -- but it lives here because it cites
`references.bib` and compiles against the `aastex701.cls` in this same
directory with no copying.  It needs one preamble line the generated paper
does not (`\usepackage{tikz}`, for its one figure); its header comment says
so.

It is the paper-facing half of a pair.  The normative half is
`src/exozippy/components/mulensing/conventions.md`, which carries the same
numbered claims `C1`-`C23` and names, for each, the source file that
implements it and the test that pins it.  A `C`-number must mean the same
thing in both files; keep edits to them in one commit.  Note that
`tests/test_prose.py`'s cite-key cross-reference scans `*.py` only, so the
keys in this `.tex` are checked by compiling it, not by the suite.
