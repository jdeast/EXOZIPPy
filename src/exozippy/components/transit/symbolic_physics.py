import sympy as sp

baseline = sp.symbols('baseline', real=True)
jitter = sp.symbols('jitter', real=True)
jitter_variance = sp.symbols('jittervar', real=True)

def get_symbol_map(config):
    return {
        "baseline": "baseline",
        "jitter": "jitter",
        "jitter_variance": "jitter_variance",
    }

RELATIONS = [
    sp.Eq(jitter_variance, jitter**2)
]

def get_solver_paths():
    return RELATIONS
