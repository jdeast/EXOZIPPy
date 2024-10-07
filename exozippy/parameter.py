import numpy as np
from astropy import units as u
import pymc as pm
import math
import ipdb

class Parameter:

    def __init__(self, label, unit=None, ndx=0,
                 initval=None,
                 expression=None,
                 lower=None, upper=None,
                 mu=None, sigma=None,
                 latex=None, latex_unit=None, description=None, latex_prefix="ez",
                 user_params=None):

        # if the user supplied a parameter file,
        # check to see if it matches the declared parameter
        l = None
        add_potential=False

        if user_params is not None:
            if label in user_params.keys():
                l = label
            elif "_0" in label:
                temp_label = label.split("_")[0]
                if temp_label in user_params.keys():
                    l = temp_label

        if l is not None:

            # if only one of initval and mu are specified
            # set initval=mu or mu=initval
            if "initval" in user_params[l]:
                initval = user_params[l]["initval"]
                if "mu" not in user_params[l]:
                    mu = initval

            if "mu" in user_params[l]:
                mu = user_params[l]["mu"]
                if "initval" not in user_params[l]:
                    initval = mu

            if "sigma" in user_params[l]:
                if user_params[l]["sigma"] == 0:
                    self.value = pm.Deterministic(label, mu)
                elif user_params[l]["sigma"] > 0:
                    if sigma is None:
                        # the user wants to impose a Gaussian penalty
                        sigma=user_params[l]["sigma"]
                    else:
                        # but there's already a gaussian penalty -- must add it as a potential
                        add_potential=True
 
            # the user can't expand the default (physical) bounds
            # only further limit them
            if "upper" in user_params[l]:
                upper = min(user_params[l]["upper"],upper)
            if "lower" in user_params[l]:
                lower = max(user_params[l]["lower"],lower)
                
        if expression is not None:
            print("deterministic: "+label+' (mu='+str(mu) +', lower='+str(lower) + ', upper='+str(upper)+', initval='+str(initval) + ',sigma='+str(sigma)+')')

            self.value = pm.Deterministic(label, expression)

            # if it's deterministic, apply constraints as potentials
            if lower is not None:
                pm.Potential("user_upperbound_" + str(ndx), pt.switch(self.value > upper, -np.inf))
            if upper is not None:
                pm.Potential("user_lowerbound_" + str(ndx), pt.switch(self.value < lower, -np.inf))
            if mu is not None and sigma is not None:
                pass
                #pm.Potential("user_prior_" + str(ndx), -0.5 * ((param.value - mu) / sigma) ** 2)
        else:
            if (mu is not None) and (sigma is not None) and ((upper is not None) or (lower is not None)):
                # bounded normal
                print("bounded normal: " + str(lower) + " < " + label + " = " + str(mu) + ' +/- ' + str(sigma) + " < " + str(upper) + " (initval="+str(initval)+")")
                self.value = pm.Truncated(label, pm.Normal.dist(mu=mu,sigma=sigma),lower=lower,upper=upper, initval=initval)
            elif (mu is not None) and (sigma is not None):
                # normal
                print("normal: " + label + " = " + str(mu) + ' +/- ' + str(sigma) + " (lower=" + str(lower)+', upper='+str(upper)+', initval='+str(initval)+')')
                self.value = pm.Normal(label, mu=mu, sigma=sigma, initval=initval)
            else:
                # uniform
                print("uniform: " + str(lower) + " < " + label + " < " + str(upper) + " (initval="+str(initval)+', mu='+str(mu)+', sigma='+str(sigma)+")")
                self.value = pm.Uniform(label, lower=lower, upper=upper, initval=initval)

        if add_potential:
            pass
            #user_prior = pm.Potential("user_prior_" + str(ndx), -0.5 * 
            #                          ((self.value - user_params[l].mu) / user_params[l].sigma) ** 2)
      
        #ipdb.set_trace()

        self.unit = unit # astropy units
        self.label = label # its unique name for specifying in the parameter file.

        # for latex table
        self.latex = latex  # latex label for the latex table
        self.latex_unit = latex_unit  # latex label for the latex table
        self.latex_value = None # $value^{+err}_{-err}$
        self.description = description  # a verbose description for the latex table
        self.latex_prefix = latex_prefix
        self.get_latex_var()
        self.table_note = None # A note to appear in the latex table

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
        new_value = ["","zero","one","two","three","four","five","six","seven","eight","nine"]
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
        if self.latex_value == None:
            self.get_latex_value()
        return "\providecommand{\\" + self.latex_varname + "}{\ensuremath{" + self.latex_value + "}}\n"

    # for median and confidence interval for latex
    def get_latex_value(self):
        if self.upper == None or self.lower==None or self.median == None:
            self.compute_confidence_interval()

        if self.upper == self.lower:
            self.latex_value = '$' + self.median + '\pm' + self.upper + '$'
        else:
            self.latex_value = '$' + self.median + '^{+' + self.upper + '}_{-' + self.lower + '}$'

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

        return "$" + self.latex + "$ \dotfill & " + self.description + unit_text + "\dotfill & " + var_txt + ' \\\\\n'

    # compute the median value, upper bound, and lower bound
    def compute_confidence_interval(self, nsigma=1.0, width=None):
        if self.posterior is not None:
            arr = np.sort(np.ndarray.flatten(self.posterior.values))
            nsamples = len(arr)
            if width==None: width = math.erf(nsigma/math.sqrt(2.0))

            lower_limit_ndx = round(width*nsamples)
            upper_limit_ndx = round(nsamples - width*nsamples)
            lowerndx = round((0.5 - width/2.0)*nsamples)
            upperndx = round((0.5 + width/2.0)*nsamples)

            medndx =  round((0.5)*nsamples)
            
            self.median_value = arr[medndx]
            self.lower_error = self.median_value - arr[lowerndx]
            self.upper_error = arr[upperndx] - self.median_value

            # round to 2 sig figs
            self.round_for_display()

        else:
            raise Exception("Cannot compute confidence interval without posteriors")

    # rounds error to two sig figs, value to match
    def round_for_display(self,sigfigs=2):
        
        nlower = -int(math.floor(math.log10(abs(self.lower_error)))) + (sigfigs-1)
        nupper = -int(math.floor(math.log10(abs(self.upper_error)))) + (sigfigs-1)

        if nlower < 0: n = min([nlower,nupper])
        else: n = max([nlower,nupper])

        self.median = str(round(self.median_value,n))
        self.lower = str(round(self.lower_error,nlower))
        self.upper = str(round(self.upper_error,nupper))
