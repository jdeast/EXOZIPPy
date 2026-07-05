# VBMicrolensing upstream PRs

Two independent patches, both applied to the local copy at
`~/python/VBMicrolensing` (uncommitted). Submit as separate PRs.

# PR 1: safedist fix — BinaryMag2 point-source shortcut never fires for rho > 1.6 when q >= 0.01

## Repo
https://github.com/valboz/VBMicrolensing
Edit: `VBMicrolensing/lib/VBMicrolensingLibrary.cpp`, `BinaryMag0` (~line 765)

## Bug
`BinaryMag2` falls back to point-source when
`corrquad < Tol && corrquad2 < 1 && safedist > 4*rho^2`.
`safedist` (distance² of the source from the caustic region of influence) is
only computed for `q < 0.01` (planetary formula); otherwise it keeps its
initialization value of **10**. Hence for any q >= 0.01 and rho > sqrt(10)/2
~ 1.6, the shortcut can never pass — the full finite-source machinery runs
even for a source thousands of Einstein radii from the lens where Mag = 1 to
machine precision, at ~0.1 s per call (~10^4 x the point-source cost; the
big-rho `BinaryMagSafe` retry cascade amplifies the constant).

Found via EXOZIPPy PTDE sampling of the Roman 2018 data challenge event 128:
hot tempering rungs proposing (s=0.1, q=2, rho=10.8) spent 56 s computing an
870-epoch light curve that is A=1 everywhere (per-call timeout rejections).

## Fix
In `BinaryMag0`, add an `else` branch for q >= 0.01: all caustics of a binary
lens lie within `R_inf = s + 1/s + 2` of the center of mass, so
`safedist = (d - R_inf)^2` for source distance d > R_inf (else keep 10).
Measured: pathological 870-epoch curve 55.7 s -> 0.119 s; 4000-point random
(s, q, x, y, rho, a1) A/B vs unpatched: max rel diff 1.5e-4 (< Tol = 1e-3),
all differences at the shortcut boundary where both answers are within
tolerance.

Note for reviewers: the same `safedist = 10` initialization affects the
multiple-lens `MultiMag2` path (no geometry-aware safedist at all); the same
bounding-circle construction applies using the SetLensGeometry positions.
Also, `BinaryMagSafe`'s "weird" heuristic `Mag * RS > 3` is unconditionally
true for RS > 3 (since Mag >= 1), triggering its most expensive retry loop
precisely when the source is enormous; `(Mag - 1) * RS > 3` preserves the
small-RS intent without that failure mode. Both are candidate follow-ups,
kept out of this PR for minimality.

# PR 2: GIL Release

## Repo
https://github.com/valboz/VBMicrolensing  
Edit: `VBMicrolensing/lib/python_bindings.cpp`

## Background
VBMicrolensing uses pybind11. Every method holds the Python GIL (Global Interpreter
Lock) throughout execution — pybind11's default. This means Python threads calling
VBMicrolensing can never overlap even when using separate C++ instances, because CPython
only runs one thread at a time while the GIL is held. For CPU-bound work like MCMC
sampling with multiple independent chains, this is a hard cap on parallelism.

No GIL-releasing code exists anywhere in `python_bindings.cpp` (confirmed by grep for
`gil_scoped_release`, `Py_BEGIN_ALLOW_THREADS`). The `pybind11/functional.h` include is
vestigial — no std::function callbacks are actually used.

## The fix
Add `py::call_guard<py::gil_scoped_release>()` to every pure-compute method. This
releases the GIL on entry to the C++ function and reacquires it on return, allowing other
Python threads to run during the (often expensive) numerical computation.

### Thread-safety requirement (must document in PR)
`VBMicrolensing` instances have per-instance mutable state (internal scratch buffers,
`NPS`, `NImages`, satellite lookup tables, etc.). GIL release is **only safe when each
thread uses its own `VBMicrolensing` instance**. Concurrent calls to a shared instance
remain a data race regardless of the GIL. This is already the natural usage pattern
(MulensModel creates one instance per model object), so no caller changes are needed.

## Methods to add `py::call_guard<py::gil_scoped_release>()` to
All pure-compute functions (numeric in → numeric out):

```
PSPLMag, ESPLMag, ESPLMag2, ESPLMagDark
BinaryMag0, BinaryMag, BinaryMagDark, BinaryMagMultiDark, BinaryMag2, BinaryMag0_shear
MultiMag0, MultiMag, MultiMagDark, MultiMag2
PSPLLightCurve, PSPLLightCurveParallax
ESPLLightCurve, ESPLLightCurveParallax
BinaryLightCurve, BinaryLightCurveW, BinaryLightCurveParallax,
  BinaryLightCurveOrbital, BinaryLightCurveKepler
BinSourceLightCurve, BinSourceLightCurveParallax, BinSourceSingleLensXallarap,
  BinSourceExtLightCurve, BinSourceExtLightCurveXallarap,
  BinSourceBinLensLightCurve, BinSourceBinLensXallarap, BinSourceLightCurveXallarap
TripleLightCurve, TripleLightCurveParallax, TripleLightCurveOrbital
LightCurve, CombineCentroids
PSPLAstroLightCurve, ESPLAstroLightCurve
BinaryAstroLightCurve, BinaryAstroLightCurveOrbital, BinaryAstroLightCurveKepler
BinSourceAstroLightCurveXallarap, BinSourceBinLensAstroLightCurve
TripleAstroLightCurve, TripleAstroLightCurveOrbital
Multicaustics, Multicriticalcurves, Caustics, Criticalcurves
ImageContours, MultiImageContours
```

## Methods to leave alone
File I/O and instance setup — calling these concurrently still requires external locking:
```
LoadESPLTable, LoadSunTable, SetESPLtablefile, SetSuntablefile, SetObjectCoordinates
```

## Code pattern

For methods bound directly via member pointer:
```cpp
// Before
vbm.def("BinaryMag2", &VBMicrolensing::BinaryMag2,
        R"(docstring...)");

// After
vbm.def("BinaryMag2", &VBMicrolensing::BinaryMag2,
        py::call_guard<py::gil_scoped_release>(),
        R"(docstring...)");
```

For lambda-wrapped methods (those that take numpy arrays and build argument lists):
```cpp
// Before
vbm.def("BinaryMag0", [](VBMicrolensing &self, ...) {
    // existing body
}, R"(docstring...)");

// After
vbm.def("BinaryMag0", [](VBMicrolensing &self, ...) {
    py::gil_scoped_release release;
    // existing body  -- note: release must be INSIDE the lambda, not as call_guard
}, R"(docstring...)");
```

`call_guard` works for non-lambda bindings. For lambdas, instantiate
`py::gil_scoped_release` as a local variable at the top of the lambda body instead.

## Test
```python
import VBMicrolensing, threading, time

def run(results, i):
    vbm = VBMicrolensing.VBMicrolensing()
    t0 = time.time()
    for _ in range(500):
        vbm.BinaryMag2(0.5, 1e-3, 0.1, 0.2, 1e-3)
    results[i] = time.time() - t0

# Serial baseline
results_serial = [None]
run(results_serial, 0)
t_serial = results_serial[0]

# 4 threads
results = [None] * 4
threads = [threading.Thread(target=run, args=(results, i)) for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
t_parallel = max(results)  # wall time = slowest thread

print(f"Serial: {t_serial:.2f}s")
print(f"4 threads wall: {t_parallel:.2f}s")
print(f"Speedup: {4 * t_serial / t_parallel:.1f}x  (ideal: 4.0x)")
```

With GIL held (before fix): speedup ≈ 1× (threads serialize).  
With GIL released (after fix): speedup ≈ 4× (threads run in parallel).
