# EXOZIPPy
[DeepWiki](https://www.deepwiki.com/jdeast/EXOZIPPy)

EXOZIPPy now has almost all features of EXOFASTv2
implemented, plus several more that EXOFASTv2 lacks. However, it is
not officially released yet. Some features are still missing and many
are not thoroughly tested. Use at your own risk and verify your
results. If you'd like to help with development, please contact me at
jason.eastman@cfa.harvard.edu, file an issue, or submit a pull
request.

## Installing

These instructions are for **using** EXOZIPPy to fit your data. If you want to
change EXOZIPPy itself, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

They are written for Linux and Apple Silicon macOS, and need no root
(`sudo`) access on Linux. On **Windows**, first follow
[`WINDOWS_INSTALL.md`](WINDOWS_INSTALL.md) to set up Linux inside Windows, then
continue here. On an **Intel Mac** (`uname -m` prints `x86_64`), follow
[`MACOS_INTEL_INSTALL.md`](MACOS_INTEL_INSTALL.md) instead.

### Step 1 -- Install Miniforge

**macOS only:** first install Apple's command line tools, which provide the
C++ compiler EXOZIPPy needs:

```bash
xcode-select --install
```

(On Linux, conda installs the compiler in Step 2.)

[Miniforge](https://github.com/conda-forge/miniforge) is a small installer for
`conda`, which gives EXOZIPPy its own Python, compiler and libraries, separate
from the system's. Skip this step if you already have `conda` (Miniforge,
Miniconda or Anaconda).

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the license, accept the default install location, and answer **yes**
when it asks whether to initialize conda. Then **close and reopen your
shell** so the `conda` command is found.

### Step 2 -- Download and install EXOZIPPy

Create the environment. On **Linux** (including WSL2):

```bash
conda create -n exozippy python=3.12 pip gxx openblas
```

On **macOS**:

```bash
conda create -n exozippy python=3.12 pip
```

EXOZIPPy compiles C code while it runs: `gxx` is the C++ compiler for that,
and `openblas` a fast linear-algebra library it links against. Keep `pip` in
the line too: recent conda versions no longer add it automatically, and
without it the `pip install` below runs some other Python's pip.

Then download and install EXOZIPPy into it:

```bash
conda activate exozippy
mkdir -p ~/python
git clone https://github.com/jdeast/EXOZIPPy.git ~/python/EXOZIPPy
cd ~/python/EXOZIPPy
pip install -e .
```

(If `git` is missing, `conda install git` provides it.) Any directory works in
place of `~/python/EXOZIPPy`; the rest of this README assumes that one.
`conda activate exozippy` is needed in every new shell before you use
EXOZIPPy.

If the compiler is ever missing or broken, EXOZIPPy checks at the start of
every fit and prints a warning naming the fix, rather than failing
mysteriously.

(EXOZIPPy is also on PyPI, as `pip install --pre exozippy`; `--pre` is
required because only pre-releases exist so far. That installs the program
but not the examples below, and the helper commands in Step 5 only exist in
releases newer than 0.1.0rc2, so the clone above is the recommended route.)

### Step 3 -- Run an example fit to check your installation

```bash
cd ~/python/EXOZIPPy/examples/hat3
exozippy hat3.yaml
```

This fits the hot Jupiter HAT-P-3b, and can take an hour or more. When it
finishes, the new `fitresults/` subdirectory holds the outputs, all starting
`HAT-P-3b_numpyro`: a `_summary.txt` and `_results.csv` of the fitted
parameters, the model plots (`_mcmc_*.pdf`), a corner plot and trace plots.
Open them with any PDF viewer, e.g. `xdg-open` on Linux, `open` on macOS, or
`explorer.exe .` under WSL2 to browse the directory in Windows File Explorer.

### Step 4 -- Fit your own system

Most example directories (`examples/*/`) have a `README.md` explaining what
that fit does. Find one (or several) that match the kind of fit you are doing
and use it as a template. Generically:

1. Collect and format your data. Keep your fits outside the repository, e.g.
   in `~/modeling/toi1234/`, so updating EXOZIPPy never touches them.
2. Define the model architecture and data sources in a `toi1234.yaml` file.
3. Define the model's starting values and priors in a `toi1234.params.yaml`
   file. Record where each prior comes from in its `citation:` field rather
   than in a comment, so it follows the prior into the parameter table (as a
   table note) and into restart files:

   ```yaml
   star.feh:
       mu: 0.27
       sigma: 0.08
       citation: "email from XX 9/9/2026"     # one citation, never split
   star.parallax:
       mu: 7.4528
       sigma: 0.0175
       citation: [GaiaCollaboration:2023, ElBadry:2021]   # or a list
   ```

   An entry that is a key in EXOZIPPy's `references.bib` is cited properly
   (`\citet`); anything else is printed as written, for you to turn into a
   citation by hand when writing the paper.
4. Run the fit:

   ```bash
   conda activate exozippy
   cd ~/modeling/toi1234
   exozippy toi1234.yaml
   ```

### Step 5 -- Helper commands for Step 4

EXOZIPPy installs three commands that do much of items 1-3 of Step 4 for you. Run each
from the directory the fit will live in (e.g. `~/modeling/toi1234`); `--help`
lists each one's options.

**Download TESS/Kepler/K2 light curves** (Gaia astrometry and Roman data are
planned):

```bash
exozippy-getdata TOI-1234
```

The argument is any SIMBAD-resolvable name. Each sector is written as its own
file (e.g. `n20190718.TESS.TESS.TOI-1234.S14.0120.SPOC.dat`).

**Build an SED file and a starting params file** from the all-sky catalogs
(broadband photometry, parallax, extinction, and starting stellar values),
which cover most exoplanet host stars:

```bash
exozippy-mkticsed TOI-1234
```

The argument is a TIC ID or any SIMBAD-resolvable name. It writes
`<name>.sed.yaml` and `<name>.params.yaml`, where `<name>` defaults to the
current directory's name (`toi1234.sed.yaml` and `toi1234.params.yaml` in
`~/modeling/toi1234`); set it with `--name`.  Note that the Gaia DR3 parallax
is corrected with Lindegren et al. (2021)'s zero-point prescription; its
uncertainty is inflated by El-Badry, Rix & Heintz (2021)'s magnitude-dependent
factor (up to 1.3x near G = 13), and then has 0.01 mas added in quadrature
for the local zero-point variations that factor leaves out; and the
photometry carries systematic error floors. Each prior it writes carries a
`citation:` naming its catalog and every correction applied.  These are best practice,
but they mean the values will not exactly match what the catalogs report. The SED file also contains many commented-out bands
that we generally do not fit, because of systematics in the source
data or in the stellar atmosphere models; uncomment them with caution.

**Convert an existing EXOFASTv2 fit:**

```bash
exozippy-exofast2exozippy ~/idl/toi1234/fittoi1234.pro
```

This reads the EXOFASTv2 driver `.pro` file and its prior and SED files, and
writes `<name>.yaml`, `<name>.params.yaml` and (if there is an SED)
`<name>.sed.yaml` into the current directory, copying the data files beside
them. `<name>` defaults to the name of the directory holding the `.pro` file
(`toi1234` here); set it with `--name`. EXOFASTv2 features with no EXOZIPPy
equivalent yet are listed as warnings at the end.

### Step 6 -- Updating EXOZIPPy

```bash
conda activate exozippy
cd ~/python/EXOZIPPy
git pull
pip install -e .
```

Re-running `pip install -e .` picks up any new or changed dependencies and
commands.

## The GUI is experimental

There is an optional browser GUI, installed by the `gui` extra
(`pip install -e ".[gui]"` in Step 2 above) and started with:

```
exozippy-gui
```

**Treat it as experimental on every platform, including Linux and macOS.** It is
still buggy and has never been verified driving a real fit end to end, so it is
not part of what EXOZIPPy supports -- unlike the CLI, nothing in CI
exercises it beyond unit tests of its own modules. Use it to look around; do not
rely on it for science.

Note this is a statement about the GUI everywhere, not a WSL caveat.
