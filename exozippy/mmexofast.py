"""
High-level functions for fitting microlensing events.
"""


class MMEXOFASTFitter():

    def __init__(self, datafiles):
        self.initial_mulens_parameters = None
        self.initial_physical_parameters = None
        self.results = None

    def get_initial_mulens_parameters(self):
        pass

    def get_initial_physical_parameters(self):
        pass

    def mcmc_fit(self):
        pass

    def fit(self):
        self.get_initial_mulens_parameters()
        self.get_initial_physical_parameters()
        self.mcmc_fit()
        return self.results
