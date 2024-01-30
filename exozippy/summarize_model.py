import pymc as pm
import numpy as np
from astropy import units as u
import astropy.constants as const
#import exoplanet as xo
import ipdb
import arviz as az
import corner
import math

def summarize_model(trace, prefix="~/modeling/test/planet.",prob=None):

    # default for astronomy is to use 1-sigma error bars
    if prob == None: prob = math.erf(1.0/math.sqrt(2.0))

    # create latex table
    summary = az.summary(trace, hdi_prob=prob)
    summary.to_latex(buf=prefix + "median.tex")

    # create pdf file
    axes = az.plot_posterior(trace,hdi_prob=prob)
    fig = axes.ravel()[0].figure
    fig.savefig(prefix + 'pdf.pdf')

    # create chain file
    chain_axes = az.plot_trace(trace)
    chain_fig = chain_axes.ravel()[0].figure
    chain_fig.savefig(prefix + 'chain.pdf')
    
    # create corner plot
#    corner_fig = corner.corner(trace)
#    corner_fig.savefig(prefix + 'corner.pdf')
