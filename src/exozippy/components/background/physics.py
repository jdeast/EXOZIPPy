from ...physics_registry import register_physics


@register_physics
def calc_background_fluxratio(f_blend, f_source, member_flux):
    """The unresolved background's flux relative to the source in a light
    curve's band: that light curve's blend ratio less the modeled stars'
    (``member_flux``, their SED-predicted fluxes relative to the source in
    the same band).  The light-curve tie of background.md, as an identity."""
    return f_blend / f_source - member_flux
