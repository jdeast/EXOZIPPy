"""
Registered physics functions for the SED component.

Everything the SED likelihood actually needs from the star (teffsed,
radiussed, logg, feh, av, distance, fbolsed, luminositysed) is already
produced by star.physics.
"""

import pytensor.tensor as pt

from ...constants import LOG_L0_CONST
from ...physics_registry import register_physics


@register_physics
def calc_absbolmag(luminosity):
    """Absolute bolometric magnitude from bolometric luminosity at d=10 pc."""
    return -2.5 * pt.log10(luminosity) + LOG_L0_CONST


@register_physics
def calc_bc(absbolmag, absmag):
    """Bolometric correction for a filter/any wavelength range"""
    return absbolmag - absmag


@register_physics
def calc_absmag_from_bc(absbolmag, bolcorr):
    """Absolute magnitude for a filter/any wavelength range using corresponding bolometric correction"""
    return absbolmag - bolcorr


@register_physics
def calc_appmag(absmag, distance):
    """Apparent magnitude using distance modulus."""
    return absmag + 5 * pt.log10(distance) - 5


@register_physics
def calc_rv_dust_from_sightline(sightline_rv_dust):
    """A star's R_V is the sight line's (sed.rv_dust): one sight line, one
    dust law (sed.md, "The spectral mode").  The context node is already
    one entry per star, so this is the identity."""
    return sightline_rv_dust
