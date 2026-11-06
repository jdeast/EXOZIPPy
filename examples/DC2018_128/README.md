This is simulated data from the 2018 Roman Data challenge of a 2L1S system. 

MMEXOFAST generated the mmexofast.json file once, and DC2018_128.params.yaml was seeded
from it (free from human intervention). The JSON is kept as the provenance record of those
starts; nothing reads it at fit time.




## The parameter file

`DC2018_128.params.yaml` carries one start per MMEXOFAST solution as a
per-seed `initval` list. With `finite_source: False`, `rho` is not a model
parameter, but its seed is not dead: the relaxation engine back-solves it
through `rho = theta_star/theta_E` into the stellar chain, so it informs
the starting `theta_E` (and through it `t_E`). Measured: removing it moved
the start's data logp by 1.3 nats through the flux decomposition. The
lens and source radius, teff and feh need no entries: with no SED, no
evolutionary model and no empirical relation nothing reads them, so they
are not parameters of this configuration; adding an `sed:` block makes
them free and constrained again with no edit to the params file.
