"""A small synthetic microlensing System for multi-source / multi-body tests.

One flat 60-epoch light curve and a configurable cast of stars, so a test
can build the topology it needs (two sources, a named stellar companion, a
per-source fitu0te flag, an event flag) without an example's data or its
params file.  The values are placeholders: these tests pin the GRAPH -- which
coordinates are sampled, which potentials exist, what the start satisfies --
not a fit.
"""

import numpy as np

from exozippy.system import System

T0 = 2458560.0


def write_flat_lc(path, n=60, span=40.0):
    """A flat 15th-magnitude light curve with mmag scatter."""
    rng = np.random.default_rng(7)
    t = np.linspace(T0 - span, T0 + span, n)
    mag = 15.0 - rng.uniform(0, 0.001, n)
    err = np.full(n, 0.01)
    np.savetxt(path, np.column_stack([t, mag, err]))
    return str(path)


# mass (solMass), distance (pc) per star name.  Lenses at 4 kpc, sources at
# 8 kpc, so pi_rel is positive and theta_E finite at the start.
_STARS = {
    "L1": (0.6, 4000.0),
    "C": (0.1, 4000.0),
    "S1": (1.0, 8000.0),
    "S2": (0.9, 8000.0),
}


def mulens_config(
    lc_path,
    *,
    lenses=("L1",),
    sources=(("S1", {}),),
    event=None,
):
    """System config: `lenses` are star names (primary first); `sources`
    are (star name, extra per-source keys) pairs; `event` extra keys go on
    the mulensevent block."""
    names = list(dict.fromkeys(list(lenses) + [s for s, _ in sources]))
    return {
        "star": [{"name": n} for n in names],
        "mulensevent": [{"name": "EV", **(event or {})}],
        "lens": [{"body": f"star.{n}"} for n in lenses],
        "source": [{"body": f"star.{n}", **kw} for n, kw in sources],
        "mulensinstrument": [{"name": "OGLE", "file": lc_path, "filter": "I"}],
    }


def mulens_params(config, extra=None):
    """Body masses/distances for every star in `config`, plus a pinned
    sky position; `extra` entries are merged last."""
    params = {
        "star.ra": {"initval": 268.0, "sigma": 0},
        "star.dec": {"initval": -29.0, "sigma": 0},
    }
    for entry in config["star"]:
        mass, dist = _STARS[entry["name"]]
        params[f"star.{entry['name']}.mass"] = {"initval": mass}
        params[f"star.{entry['name']}.distance"] = {"initval": dist}
    params.update(extra or {})
    return params


def build(config, params):
    """prepare() + build_model(); returns (system, model)."""
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    return system, model
