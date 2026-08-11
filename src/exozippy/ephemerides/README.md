This directory contains ephemeris files to compute the observer trajectory (microlensing parallax, astrometry parallax factors, etc.; see src/exozippy/ephemeris.py). New ephemerides can be generated in the appropriate format using the get_ephemeris tool in this directory. This is how the existing ones were generated:

python get_ephemeris.py --id -79 --out "spitzer.eph" --start "2013-06-01" --stop "2018-12-31" --step="1d" 
python get_ephemeris.py --id -163 --out "k2.eph" --start "2016-04-01" --stop "2016-07-31" --step="1h" 

See here for a list of IDs:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html

roman_simulated_2018dc.eph
  Source: MMEXOFAST 2018DataChallenge/wfirst_ephemeris_W149.txt
  https://github.com/jenniferyee/MMEXOFAST
  Format: BJD X_EQ Y_EQ Z_EQ X_ECL Y_ECL Z_ECL (AU, J2000 equatorial + ecliptic)
  Coverage: BJD 2458346.505 – 2460059.241 (~4.7 yr at 15-min cadence)
  Columns 1-3 (X_EQ, Y_EQ, Z_EQ) are the equatorial J2000 frame used by EXOZIPPy.
  Single file covers all 2018 DC filters (spacecraft position is filter-independent).
gaia.eph
  Source: JPL Horizons (ID -139479), generated with:
  python get_ephemeris.py --id "-139479" --out "gaia.eph" --start "2014-01-01" --stop "2025-03-01" --step="1d"
  Coverage: BJD 2456658.5 - 2460735.5 (full Gaia science mission at L2)
