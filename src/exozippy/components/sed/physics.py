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


@register_physics
def calc_av_from_profile(profile_av, av_frac):
    """An object's A_V under the sight-line dust profile: the profile at its
    distance, A(d) (a context node), times its own arcsec-scale scatter
    about it, 1 + av_frac (sed.md, "The sight-line dust profile")."""
    return profile_av * (1.0 + av_frac)


@register_physics
def calc_rv_dust_from_profile(profile_rv_dust):
    """An object's R_V under the near/far split: the A_V-weighted mix of the
    two R_V values in front of it, 1/R = f/R_near + (1 - f)/R_far (a context
    node, one entry per element)."""
    return profile_rv_dust


@register_physics
def calc_sed_anchor_av(anchor_av):
    """A_V at the dust anchor on the reference spectrum, from the anchor
    band's extinction (a context node: the band-integrated A_band(A_V)
    inverted on the SED's own grid)."""
    return anchor_av
