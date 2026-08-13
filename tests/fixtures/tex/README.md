# Vendored TeX class files

## `aastex701.cls`

AASTeX v7.0.1, 2025-05-09. Copyright 2025 American Astronomical Society,
distributed under the **LaTeX Project Public License 1.3c** (the header of
the file itself carries the licence), so redistributing it here is allowed.

- Source: <https://mirrors.ctan.org/macros/latex/contrib/aastex.zip>, file
  `aastex/aastex701.cls`
- Size: 383761 bytes
- sha256: `c1bb436807f8bb37eb5203056db5a5a60e1c9ae6a3feb6501d7c414b9b08b062`

### Why it is in the repo rather than a setup step

`tests/test_latex_macro_xref.py` compiles the deluxetable EXOZIPPy generates
as a real AASTeX document, which is the only check that catches a table body
that is valid-looking but not valid LaTeX -- a stray unescaped `%`, an `&`
count that disagrees with the column spec, a `\tablecomments{}` that swallows
its line. Typesetting the cited macros in a minimal `article` (the test above
it) proves the macro *names* resolve and nothing more.

Asking every developer to install AASTeX to get that would mean almost nobody
ran it. At ~375 kB the class file is smaller than several fixtures already
here, and it makes the check work out of the box for anyone who has any TeX
at all. `TEXINPUTS` points at this directory for the duration of the test;
nothing is installed into a TeX tree.

The test skips when `pdflatex` is absent, so no TeX installation is required
to run the suite -- which is why CI does not exercise it.

### To update

Refetch from the URL above, replace the file, and update the size and hash
here. `\documentclass{aastex701}` in `outputs/latex.py` names the version, so
moving to a later AASTeX means changing both.

**Do not add `apjfonts.sty`.** The generated template used to load it; it is a
legacy AASTeX v5 font package, it is on neither CTAN nor TeX Live, and its
absence is a fatal `Emergency stop` rather than a degraded render -- so that
one line made the generated file uncompilable for anyone who did not happen to
already have it. AASTeX 7 sets its own fonts and AAS's own `aastex701-sample.
tex` does not load it. See the comment in `outputs/latex.py`.
