"""Write the simulated example event in input_data/ and input_events/.

A point-source point-lens event seen by Roman in F146, one 72-day bulge
season at 1-hour cadence, placed inside the coverage of the shipped
roman_simulated_2018dc ephemeris (the 2018 Data Challenge orbit; Roman's
real ephemeris is not shipped yet).  Rerun to regenerate:

    python make_example_event.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T0, U0, TE = 2458620.0, 0.15, 25.0
BASE_MAG, FS_FRAC = 20.0, 0.7
SIGMA_MAG = 0.01


def main():
    rng = np.random.default_rng(20261105)
    t = np.arange(T0 - 36.0, T0 + 36.0, 1.0 / 24.0)
    u = np.hypot(U0, (t - T0) / TE)
    amp = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    flux = FS_FRAC * amp + (1 - FS_FRAC)
    mag = BASE_MAG - 2.5 * np.log10(flux) + rng.normal(0, SIGMA_MAG, t.size)
    err = np.full(t.size, SIGMA_MAG)
    (HERE / "input_data").mkdir(exist_ok=True)
    np.savetxt(
        HERE / "input_data" / "sim_pspl_F146.dat",
        np.column_stack([t, mag, err]),
        fmt=["%.6f", "%.5f", "%.4f"],
        header="BJD_TDB  F146_mag  err  (simulated PSPL: "
        f"t_0={T0} u_0={U0} t_E={TE})",
    )
    event = {
        "event_id": "sim_pspl_0001",
        "ra": 268.0,
        "dec": -29.0,
        "architecture": "1S1L",
        "light_curves": [
            {
                "path": "sim_pspl_F146.dat",
                "band": "F146",
                "data_format": "magnitude",
                "observer_location": "roman_simulated_2018dc",
                "time_frame": "bjd",
                "time_scale": "tdb",
            }
        ],
    }
    (HERE / "input_events").mkdir(exist_ok=True)
    with open(HERE / "input_events" / "sim_pspl_0001.json", "w") as f:
        json.dump(event, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
