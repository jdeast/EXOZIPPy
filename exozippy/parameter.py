import numpy as np


class Parameter:

    def __init__(self, label, value=None):
        self.value = None  # its value
        self.posterior = None  # NSTEPS x NCHAINS array of all values
        self.prior = None  # its prior value
        self.gaussian_width = np.inf  # N(prior,gaussian_width)
        self.lowerbound = -np.inf  # U(lowerbound,upperbound) ## do we want soft edges?
        self.upperbound = np.inf  # U(lowerbound,upperbound)
        self.label = None  # its unique name for specifying in the parameter file.
        self.cgs = None  # multiply by this factor to convert to cgs units
        self.link = None  # a pointer(?) to another parameter
        self.amoeba_scale = None  # scale for amoeba steps (roughly the range it will explore)

        # for latex table
        self.latex = ''  # latex label for the latex table
        self.description = ''  # a verbose description for the latex table

        # do we want to use the unit package? What kind of overhead does that impose?
        self.unit = None  # units ("Radians" triggers special angular behavior)

        self.userchanged = False  # the user supplied a prior that impacts this parameter
        self.fit = False  # If true, step in this parameter during the MCMC
        self.derive = True  # If True, quote this parameter in the final table

        self.medvalue = ''  # median of posterior after trimming burn-in
        self.upper = ''  # upper error bar (68% confidence)
        self.lower = ''  # lower error bar (68% confidence)
        self.best = None  # value corresponding to the highest likelihood solution

    # compute the median value, upper bound, and lower bound
    def compute_confidence_interval(self):
        pass

    # rounds error to two sig figs, value to match
    def round_for_display(self):
        pass
