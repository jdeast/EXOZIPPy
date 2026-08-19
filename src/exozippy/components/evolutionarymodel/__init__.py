# Deliberate landing pad for the evolutionarymodel component (MIST, PARSEC,
# YY) -- review item 8.8.4.  Empty on purpose; do NOT delete it (review
# 5.8.2).
#
# It is empty and still correct because `factory.discover_components` skips
# `__init__.py` and finds components by scanning for `Component` subclasses in
# the OTHER modules of each subdirectory, so a directory with no such module
# registers nothing and no YAML key `evolutionarymodel:` resolves to a
# component.  A config naming one today gets System's "does not match any
# registered component" warning, plus a sharper one from
# `Star.register_parameters` about the likelihood-free track coordinates it
# would create.
#
# What lands here is the component's four standard files
# (`evolutionarymodel.py`, `defaults.yaml`, `symbolic_physics.py`,
# `physics.py`; see `src/exozippy/components/components.md`).  The STAR SIDE
# IS ALREADY DONE and must not need touching: `star/defaults.yaml` carries
# `age`, `initfeh` and `eep` with MIST's own bounds, the
# `in_topology("evolutionarymodel")` branch of `Star.register_parameters`
# declares all three, and the per-star `mist:`/`parsec:` switches are a real
# per-element activity mask -- a star with no track has no EEP.  The
# reasoning, including why `eep` is spelled "Equivalent Evolutionary Point",
# is in `src/exozippy/components/star/star.md`.
#
# The grids and the scripts that build them are already in the tree, under
# `src/exozippy/models/MIST/`.
