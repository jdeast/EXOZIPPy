# EXOZIPPy
[DeepWiki](https://www.deepwiki.com/jdeast/EXOZIPPy)

initial commit for EXOFASTv2 python translation
This will eventually be a python implementation of EXOFASTv2, but is a long ways off

When we have a functional stub, it'll be a pip installable package. For now, install it like this:

```
git clone https://github.com/jdeast/EXOZIPPy.git
pip install .
```

EXOZIPPY_PATH environment variable?

If you'd like to help, please contact me at jason.eastman@cfa.harvard.edu

# Conda installation
conda create -n exozippy python=3.11
conda activate exozippy
conda install -c conda-forge pymc pytensor arviz numpy scipy pandas matplotlib astropy corner
pip install -e .

# pip/venv
python -m venv venv
source venv/bin/activate  # (or Activate.ps1)
pip install -U pip
pip install -e .

# Windows (powershell) -- skips compiler (slow!) Installation via Conda recommended!
setx PYTENSOR_FLAGS "blas__ldflags=,cxx="
