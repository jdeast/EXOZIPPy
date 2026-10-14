import numpy as np
import scipy.stats as stats


class Contour:
    def __init__(
        self,
        x,
        y,
        x_err,
        y_err,
        probs,
        bw_method=0.7,
    ):
        """A Gaussian-KDE density of the draws ``(x, y)`` on a 100 x 100
        grid reaching 3 ``x_err`` / ``y_err`` past them, and the density
        levels enclosing ``probs`` of it (highest-density regions).

        ``probs`` are the enclosed probabilities to contour, largest first,
        so ``levels`` comes out in the increasing density order ``contour``
        wants; no default, because each figure states its own ladder.
        ``bw_method`` is ``scipy.stats.gaussian_kde``'s (a factor, or
        "scott"/"silverman"); the 0.7 default is the evolutionary model's
        contour plot's.  Drawing is the caller's: the
        evolutionary model's contour PDF
        (``components/evolutionarymodel/plot.py``) and the summary figure
        each style their own."""
        self.x = x
        self.y = y
        self.x_err = x_err
        self.y_err = y_err
        self.bw_method = bw_method
        self.probs = tuple(probs)

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
        kernel = stats.gaussian_kde(values, bw_method=self.bw_method)
        Z = np.reshape(kernel(positions).T, X.shape)

        ###### 3. Calculate contour levels corresponding to specific confidence intervals
        ###### We sort density values to find thresholds matching 68% and 95% of total mass
        # global fit values
        z_sorted = np.sort(Z.ravel())
        z_cumulative = np.cumsum(z_sorted) / np.sum(z_sorted)

        # The density threshold enclosing each probability (the highest-
        # density region), widest first.
        levels = [
            z_sorted[np.searchsorted(z_cumulative, 1.0 - p)]
            for p in self.probs
        ]

        return X, Y, Z, levels
