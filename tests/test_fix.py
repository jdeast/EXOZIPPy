import pymc as pm
import pytensor
import numpy as np


def test_fresh_start():
    # Setup global state to trigger the error if possible
    # (In your case, Windows does this automatically)

    def build_logic():
        x = pm.Normal('x', 0, 1, shape=100)
        mu = pm.math.sum(x)
        pm.Normal('y', mu=mu, sigma=1, observed=0.5)

    try:
        with pm.Model() as model1:
            build_logic()
            pm.sample(draws=1, tune=1)
    except Exception:
        print("\nCaught failure. Switching to Safety Mode and rebuilding...")

        # Apply global settings
        pytensor.config.cxx = ""
        pytensor.config.linker = "vm"

        # REBUILD FRESH
        with pm.Model() as model2:
            build_logic()
            print("Sampling fresh model...")
            pm.sample(draws=5, tune=5, cores=1)
            print("SUCCESS!")


if __name__ == "__main__":
    test_fresh_start()