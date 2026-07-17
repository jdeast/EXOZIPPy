# EXOZIPPy
[DeepWiki](https://www.deepwiki.com/jdeast/EXOZIPPy)

This will eventually be a python successor to EXOFASTv2, but it is not officially released yet. Many features are missing, not tested, or not functional. If you'd like to help with development, please contact me at jason.eastman@cfa.harvard.edu

When we officially release the code, it'll be a pip installable package. For now,
install it from a git clone. All dependencies (including exoplanet-core, which now
ships binary wheels) resolve from PyPI, so no compiler is required:

```
conda create -n exozippy python=3.12
conda activate exozippy
mkdir -p ~/python
cd ~/python
git clone https://github.com/jdeast/EXOZIPPy.git
cd EXOZIPPy
pip install -e .
```

If you have Poetry, `poetry install` (which uses the pinned `poetry.lock`) is the
most reliable path.

Note: exoplanet-core provides wheels for CPython 3.12-3.14 on Linux (glibc 2.28+),
Apple Silicon macOS, and Windows. Intel macOS has no wheel and builds from source,
which needs a C++ compiler.

# Windows (powershell)
good luck! If you manage to install it and run it on windows, please send instructions.

skips compiler (slow!) Installation via Conda recommended!

This might be relevant:
setx PYTENSOR_FLAGS "blas__ldflags=,cxx="
