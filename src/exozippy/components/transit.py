# this is mostly a placeholder
class Transit:
    def __init__(self, name, data):
        self.name = name
        self.data = data  # Contains the time, rv, err arrays

    def build_parameters(self, model):
        prefix = f"instrument.{self.name}"

        gamma_init = np.mean(self.data.rv)
        # Use your hard-coded default (5.0 m/s) unless overridden
        gamma_scale = 5.0

        self.gamma = pm.Normal(f"{prefix}.gamma", mu=gamma_init, sigma=100.0)
        self.jitter = pm.HalfNormal(f"{prefix}.jitter", sigma=10.0)
        return {"gamma": self.gamma, "jitter": self.jitter}

    def load_data(self, path):
        self.data = np.loadtxt(path,unpack=True)