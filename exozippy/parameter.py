import numpy as np
from astropy import units as u
import pymc as pm

class Parameter:

    def __init__(self, label, unit=None,
                 initval=None,
                 Deterministic=False, expression=None,
                 Uniform=True, lower=None, upper=None,
                 Normal=False, mu=None, sigma=None,
                 latex=None, latex_unit=None, description=None, latex_prefix="ez@"):

        # its pytensor distribution
        if Deterministic:
            self.value = pm.Deterministic(label, expression)
        elif Normal:
            self.value = pm.Normal(label, mu=mu, sigma=sigma)
        elif Uniform:
            self.value = pm.Uniform(label, lower=lower, upper=upper, initval=initval)

        self.unit = unit # astropy units
        self.label = label # its unique name for specifying in the parameter file.

        # for latex table
        self.latex = latex  # latex label for the latex table
        self.latex_unit = latex_unit  # latex label for the latex table
        self.latex_value = None # $value^{+err}_{-err}$
        self.description = description  # a verbose description for the latex table
        self.latex_prefix = latex_prefix
        self.get_latex_var()

        self.posterior = None  # NSTEPS x NCHAINS array of all values
        self.prior = None  # its prior value
        self.gaussian_width = np.inf  # N(prior,gaussian_width)
        self.lowerbound = -np.inf  # U(lowerbound,upperbound) ## do we want soft edges?
        self.upperbound = np.inf  # U(lowerbound,upperbound)
        self.link = None  # a pointer(?) to another parameter
        self.amoeba_scale = None  # scale for amoeba steps (roughly the range it will explore)

        self.userchanged = False  # the user supplied a prior that impacts this parameter
        self.fit = False  # If true, step in this parameter during the MCMC
        self.derive = True  # If True, quote this parameter in the final table

        self.medvalue = None  # median of posterior after trimming burn-in
        self.upper = None  # upper error bar (68% confidence)
        self.lower = None  # lower error bar (68% confidence)
        self.best = None  # value corresponding to the highest likelihood solution

    def to_unit(self, target):
        return self.value * self.unit.to(target)

    '''
    Because "_" is not allowed in latex variable names, we replace it with @
    Because numbers are not allowed in latex variable names, we replace them with their english spelling
    Because latex variable names are global in scope, we add preface all variable names with "ez@"
    '''
    def get_latex_var(self):
        # make the label a legal latex variable name
        old_value = ["_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        new_value = ["@","zero","one","two","three","four","five","six","seven","eight","nine"]
        varname = self.label
        for i in range(len(old_value)): varname = varname.replace(old_value[i],new_value[i])
        self.latex_varname = self.latex_prefix + varname

    ''' returns a string that defines a latex variable equal to the numerical value with errors
    \providecommand{\varname}{\ensuremath{value^{+err}_{-err}}}'
    
    These definitions will be output into a separate values.tex file. We encourage exozippy users to use these variable 
    names throughout their manuscript to avoid leaving stale values after updating their fits. 
    
    Build custom tables that will be updated with a new run just by replacing this file. 
    
    Templates can be reused for consistent style between different systems.     
    '''
    def to_latex_var(self):
        return "\providecommand{\\" + self.latex_varname + "}{\ensuremath{" + self.latex_value + "}}"

    # format a line in a latex table
    # $symbol$ ... description (units) ... $value^{+upper}_{-lower}$\\
    def to_table_line(self, use_variable=True):
        if self.latex_unit == None:
            unit_txt = ''
        else:
            unit_text = " (" + self.latex_unit + ")"
        if use_variable:
            var_txt = "\\" + self.latex_varname
        else:
            var_txt = "\ensuremath{" + self.latex_value + "}"

        return "$" + self.latex + "$ \dotfill & " + self.description + unit_text + "\dotfill & " + var_txt

    # compute the median value, upper bound, and lower bound
    def compute_confidence_interval(self):
        pass

    # rounds error to two sig figs, value to match
    def round_for_display(self):
        pass
