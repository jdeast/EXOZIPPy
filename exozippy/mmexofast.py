"""
High-level functions for fitting microlensing events.
"""


class MMEXOFASTFitter():

    def __init__(self, datafiles):
        pass

    def get_init_ulens_params(self):
        pass

    def get_phys_params(self):
        pass

    def mcmc_fit(self):
        pass

    def fit(self):
        init_ulens_params = get_init_ulens_params(filelist)
        init_phys_params = get_phys_params(init_ulens_params)
        results = mcmc_fit(filelist, init_phys_params)
