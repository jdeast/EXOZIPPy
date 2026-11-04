# Cross-checking the barycentric correction against pulsar-timing codes

How the in-model barycentric correction of `rvinstrument` `data_format: raw_redshift`
(`src/exozippy/components/rvinstrument/redshift.py`) was compared with TEMPO, TEMPO2 and
PINT, so that anyone can repeat it. The results are in
`src/exozippy/components/rvinstrument/rvinstrument.md`, "Validation". None of these codes
is an EXOZIPPy dependency; install them in a throwaway environment.

## Installing the codes

- **TEMPO (TEMPO1) 13.103** (git `13.100-44-gaaf34c4`): build it from the SourceForge git repository
  (`git clone https://git.code.sf.net/p/tempo/tempo`, then `./prepare && ./configure &&
  make`). The build takes about a minute. Make a private runtime directory (`$TEMPO`)
  holding a `tempo.cfg`, the `clock/` tree and an `obsys.dat` with your sites' ITRF XYZ.
  The shipped `ut1.dat` stops in 2021, so regenerate it from an IERS EOP C04 file (for
  example the one TEMPO2 ships). TEMPO1 has no polar-motion model.
- **TEMPO2 and PINT** from conda-forge, into their own environment:
  `conda create -p <scratch>/tempo-env -c conda-forge tempo2 pint-pulsar` (the comparison
  used tempo2 2026.04.1 and pint-pulsar 1.1.8).

## The method

The method is Wright & Eastman (2014) Appendix A's, applied to all three codes. Each
(site, star, UTC epoch) is set up as a "pulsar":

- the star is re-referenced to that day (POSEPOCH = PEPOCH = the day, astrometry
  propagated with `erfa.pmsafe`);
- F0 = 1 Hz, F1 = 0, DM = 0.

The topocentric polyco frequency is then `nu_obs / nu_SSB`, the barycentric correction
with nothing of the star left in it. Re-referencing per day keeps each code inside its
exact regime. The star's secular-acceleration factor is applied afterwards,
analytically.

Native TEMPO2 (not `-tempo1` prediction mode) and PINT use IAU 2006/2000A and polar
motion, as EXOZIPPy does. TEMPO1 and TEMPO2's `-tempo1` polyco mode omit polar motion,
which is the sidereal-day residual Wright & Eastman reported.

## Floating-point floor

A single float64 Julian Date quantizes at ~40 microseconds. That puts a floor of 0.0004
mm/s RMS on any comparison at a ground site, which comes from the Earth's rotational
acceleration. No quad-precision time path is needed at this level.
