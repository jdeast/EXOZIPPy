https://ui.adsabs.harvard.edu/abs/2014ApJ...795...42P/abstract

While this system is complex (binary star + Uranus), we're modeling just the late time event with a simple PSPL fit, using our native pytensor Paczyński curve suitable for fast NUTS steps.


## The parameter file

`ob08092.params.yaml` seeds the trajectory from Poleski et al. (2014),
Table 1 (the secondary subevent), and pins the event coordinates. The
lens and source kinematics (distances, proper motions, rv) are left to
the galactic model and the coordinates. This is a point-source fit with no
SED, so nothing reads the stellar radius, teff or feh: those entries are
start values only (`star.Source.radius = 0.519` is Poleski et al.'s, the
lens values a generic M dwarf), and adding an `sed:` block makes them
free, constrained parameters again with no edit to the params file.
