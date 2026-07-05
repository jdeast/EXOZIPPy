# EXOZIPPy
[DeepWiki](https://www.deepwiki.com/jdeast/EXOZIPPy)

This will eventually be a python successor to EXOFASTv2, but it is not officially released yet. Many features are missing, not tested, or not functional. If you'd like to help with development, please contact me at jason.eastman@cfa.harvard.edu

When we officially release the code, it'll be a pip installable package. For now, install it with conda (PyMC 5 is a heavy dependency and can only reliably be installed with conda):

```
conda create -n exozippy python=3.12
conda activate exozippy
conda install -c conda-forge pymc pytensor arviz numpy scipy pandas matplotlib astropy corner
mkdir -p ~/python
cd ~/python
git clone https://github.com/jdeast/EXOZIPPy.git
cd EXOZIPPy
pip install -e .
```

# Windows (powershell)
good luck! If you manage to install it and run it on windows, please send instructions.

skips compiler (slow!) Installation via Conda recommended!

This might be relevant:
setx PYTENSOR_FLAGS "blas__ldflags=,cxx="
