This directory contains ephemeris files to compute the observer's trajectory during a microlensing event. New ephemerides can be generated in the appropriate format using the get_ephemeris tool in this directory. This is how the existing ones were generated:

python get_ephemeris.py --id -79 --out "spitzer.eph" --start "2013-06-01" --stop "2018-12-31" --step="1d" 
python get_ephemeris.py --id -163 --out "k2.eph" --start "2016-04-01" --stop "2016-07-31" --step="1h" 

See here for a list of IDs:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html