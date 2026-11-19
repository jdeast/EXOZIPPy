"""The SED figure for ``model: PHOENIX``.

The fit's bolometric corrections come from the PHOENIX tables, but no
plotting-resolution PHOENIX spectra are published, so the figure's model
curve is drawn from the packaged NextGen spectra: ``components/sed/
plot.Plot`` finds no ``PHOENIX.spectra.csv`` beside this file, falls back
to NextGen and says so in a WARNING.  The photometry, residuals and the
predicted band magnitudes are the fit's own.
"""

from exozippy.models.NextGen.BCs.plot import NextGenPlot


class PHOENIXPlot(NextGenPlot):
    """NextGenPlot under the PHOENIX model's name (see the module
    docstring)."""
