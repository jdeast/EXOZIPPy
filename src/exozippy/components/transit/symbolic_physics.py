import sympy as sp

baseline = sp.symbols("baseline", real=True)
jitter = sp.symbols("jitter", real=True)
jitter_variance = sp.symbols("jittervar", real=True)


def get_symbol_map(config):
    return {
        "baseline": "baseline",
        "jitter": "jitter",
        "jitter_variance": "jitter_variance",
    }


# Reparameterization bridge: the user may provide 'jitter' while the sampler
# steps in 'jitter_variance'.  Signed square, matching the signed square root
# the model reports (components/instrument.py:calc_jitter) over the negative
# variances Instrument._jitter_floor deliberately allows; jitter**2 would fold
# a negative seed onto a positive variance.
#
# NOTE (pre-existing, deliberately not changed here): the sympy symbol above
# is named "jittervar" while get_symbol_map's key is "jitter_variance".  The
# ConfigManager substitutes relation symbols by sym.name, so this symbol is
# never bound and the relation is inert for transit.  Renaming it would make
# the relation live and change transit start values, which is a separate
# change from allowing negative jitter.
RELATIONS = [sp.Eq(jitter_variance, jitter * sp.Abs(jitter))]


def get_solver_paths():
    return RELATIONS
