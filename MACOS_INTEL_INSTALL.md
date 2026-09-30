# Running EXOZIPPy on an Intel Mac (macOS x86_64)

Intel Macs need one extra install step, and two features are unavailable.
Check which Mac you have with `uname -m`: `x86_64` is Intel; `arm64`
(Apple Silicon) needs none of this, so follow [`README.md`](README.md)
directly.

What does not work on an Intel Mac:

- the `gp:` key (Gaussian-process noise);
- the `numpyro` and `blackjax` samplers. Use `ptde` (the default), `nuts`
  or `nutpie`.

Everything else (RV, transit, SED, astrometry, microlensing) works
normally. The cause is upstream: no current version of jax can be
installed on Intel macOS.

## Step 1 -- Follow the README's Step 1

Do Step 1 of "Installing" in [`README.md`](README.md) (the Xcode command
line tools and Miniforge). The Xcode compiler matters more here than
elsewhere: some dependencies have no Intel Mac build and are compiled
during the install.

## Step 2 -- Download EXOZIPPy and create the environment

Use Python **3.12 or 3.13**; 3.14 will not install on an Intel Mac.

```bash
mkdir -p ~/python
git clone https://github.com/jdeast/EXOZIPPy.git ~/python/EXOZIPPy
conda create -n exozippy python=3.12 pip
conda activate exozippy
```

## Step 3 -- Build celerite2 first

This must come **before** `pip install -e .`; without it, that install
fails with `Cannot install build-system.requires for celerite2`.

```bash
pip install scikit-build-core "numpy>=2.0,<2.4" pybind11 cmake ninja setuptools setuptools_scm
pip install --no-build-isolation \
    --config-settings=cmake.define.BUILD_JAX=OFF \
    "celerite2>=0.3.3,<0.4.0"
```

This step compiles, and is slow. It will no longer be needed once
[celerite2#193](https://github.com/exoplanet-dev/celerite2/pull/193) is
released.

## Step 4 -- Install EXOZIPPy and continue with the README

```bash
cd ~/python/EXOZIPPy
pip install -e .
```

Then continue from Step 4 of "Installing" in [`README.md`](README.md).

The README's Step 3 example uses the `numpyro` sampler, which is
unavailable here. Run a copy of it with the default sampler instead
(copying keeps the repository unmodified, so `git pull` stays clean):

```bash
mkdir -p ~/modeling
cp -r ~/python/EXOZIPPy/examples/hat3 ~/modeling/hat3
cd ~/modeling/hat3
sed -i '' 's/method: numpyro/method: ptde/' hat3.yaml
exozippy hat3.yaml
```

What to expect from the fit is described in the README's Step 3.

If you will be developing EXOZIPPy rather than only running fits,
[`CONTRIBUTING.md`](CONTRIBUTING.md) covers the Intel Mac differences for
developers.
