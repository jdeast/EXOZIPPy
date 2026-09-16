import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.colors import to_rgba


class Contour:
    def __init__(self, x, y, x_err, y_err, **kwargs):
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.X, self.Y, self.Z, self.levels = self._generate_contours()

    def _generate_contours(self):
        """Generate contours for the Kiel diagram based on the posterior distribution of stellar parameters.

        This function generates  with contours representing the
        posterior distribution of the stellar parameters. It uses kernel density
        estimation (KDE) to estimate the density of points in the Kiel diagram.

        Parameters
        ----------
            x : array-like
                The x-coordinates of the data points.
            y : array-like
                The y-coordinates of the data points.
            x_err : array-like
                The errors in the x-coordinates.
            y_err : array-like
                The errors in the y-coordinates.

        Returns
        -------

        """
        ##### 1. Set up the grid for KDE evaluation
        xmin, xmax = (
            self.x.min() - 3 * self.x_err,
            self.x.max() + 3 * self.x_err,
        )
        ymin, ymax = (
            self.y.min() - 3 * self.y_err,
            self.y.max() + 3 * self.y_err,
        )
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])

        ##### 2. Fit a Kernel Density Estimation (KDE)
        # global fit values
        values = np.vstack([self.x, self.y])
        kernel = stats.gaussian_kde(values, bw_method=0.7)
        Z = np.reshape(kernel(positions).T, X.shape)

        ###### 3. Calculate contour levels corresponding to specific confidence intervals
        ###### We sort density values to find thresholds matching 68% and 95% of total mass
        # global fit values
        z_sorted = np.sort(Z.ravel())
        z_cumulative = np.cumsum(z_sorted) / np.sum(z_sorted)

        # Find the density thresholds for 68% and 95% confidence regions
        level_68 = z_sorted[np.searchsorted(z_cumulative, 1.0 - 0.68)]
        level_95 = z_sorted[np.searchsorted(z_cumulative, 1.0 - 0.95)]
        level_99 = z_sorted[np.searchsorted(z_cumulative, 1.0 - 0.997)]
        levels = [level_99, level_95, level_68]

        return X, Y, Z, levels


def plot_contours(
    groups,
    xlabel=r"$T_{\rm{eff}}$ (K)",
    ylabel=r"$\log g$ (cgs)",
    figsize=(7, 6),
):

    fig, ax = plt.subplots(figsize=figsize)
    _xmin, _xmax, _ymin, _ymax = None, None, None, None
    _xerr, _yerr = None, None

    for g in groups:
        if isinstance(g, Contour):
            color = g.color if hasattr(g, "color") else "blue"
            alpha = g.alpha if hasattr(g, "alpha") else 0.05
            linewidth = g.linewidth if hasattr(g, "linewidth") else 2
            size = g.size if hasattr(g, "size") else 35
            label = g.label if hasattr(g, "label") else None

            ax.scatter(
                g.x,
                g.y,
                s=size,
                color=to_rgba(color, alpha),
                label=label,
                zorder=3,
            )

            contour_alphas = [
                0.25,
                0.5,
                0.75,
            ]  # Different alpha values for different confidence levels
            rgba_colors_contours = [
                to_rgba(color, alpha=a) for a in contour_alphas
            ]

            cf = ax.contourf(
                g.X,
                g.Y,
                g.Z,
                levels=g.levels + [g.Z.max()],
                colors=rgba_colors_contours,
            )
            cl = ax.contour(
                g.X,
                g.Y,
                g.Z,
                levels=g.levels,
                colors=rgba_colors_contours,
                linewidths=linewidth,
            )

            # update the axis limits based on the data
            _xmin = g.x.min() if _xmin is None else min(_xmin, g.x.min())
            _xmax = g.x.max() if _xmax is None else max(_xmax, g.x.max())
            _ymin = g.y.min() if _ymin is None else min(_ymin, g.y.min())
            _ymax = g.y.max() if _ymax is None else max(_ymax, g.y.max())

            # update the axis limits based on the errors
            _xerr = g.x_err if _xerr is None else max(_xerr, g.x_err)
            _yerr = g.y_err if _yerr is None else max(_yerr, g.y_err)

    # Set axis limits
    ax.set_xlim(_xmin - 3 * _xerr, _xmax + 3 * _xerr)
    ax.set_ylim(_ymin - 3 * _yerr, _ymax + 3 * _yerr)

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend(loc="upper left")

    for handle in leg.legend_handles:
        handle.set_alpha(0.8)

    ax.invert_xaxis()
    ax.invert_yaxis()

    return fig, ax
