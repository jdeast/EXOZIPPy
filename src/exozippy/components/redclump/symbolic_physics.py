comp_key = "redclump"

# The clump's sampled coordinates are relaxation-engine leaves, so its starts
# are owned like every other parameter's (a user may seed any of them, a
# stage-3 solver may probe them).  Its radius, (V-I)_0 and band extinctions
# read the SED's bolometric-correction grid, which is not symbolic: they are
# in no relation.
RELATIONS = []


def get_symbol_map(component_config, system_config):
    return {
        name: name for name in ("teff", "logg", "feh", "mi", "distance", "av")
    }
