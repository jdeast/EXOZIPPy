"""
Analyze the microlensing event OB140939 and produce all outputs for the paper.
"""
import exozippy
import os.path
import matplotlib.pyplot as plt


"""
Desired outputs:

Microlensing:
- Static PSPL model parameters & uncertainties, chi2
- Parallax PSPL model parameters & uncertainties, chi2
- Parallax contour plot

EXOZIPPy:
- Best parallax PSPL model parameters & uncertainties, chi2
- Posteriors for physical parameters

Note: all chi2s need to be on the same system.
"""

ground_data_files = [os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB140939', 'n20100310.I.OGLE.OB140939.txt')]
space_data_files = [os.path.join(
        exozippy.MULENS_DATA_PATH, 'OB140939', 'n20100310.I.OGLE.OB140939.txt')]
coords='17:47:12.25 -21:22:58.7'


class ParallaxFitter():

    def __init__(self, data_files=None, coords=None, fit_type=None):
        self.data_files = data_files
        self.coords = coords
        self.fit_type = fit_type

        self._fitter = None

    def fit(self, **kwargs):
        self.fitter = exozippy.mmexofast.MMEXOFASTFitter(
            files=self.data_files, coords=self.coords, fit_type=self.fit_type, **kwargs)
        self.fitter.fit()

    def print_latex_results_table(self):
        print(self.fitter.make_ulens_table(type='latex'))

    def plot_lc(self):
        pass

    def plot_parallax_grids(self):
        pass

    @property
    def fitter(self):
        return self._fitter

    @fitter.setter
    def fitter(self, value):
        self._fitter = value


class SpaceParallaxFitter(ParallaxFitter):

    def __init__(self, ground_data_files=None, space_data_files=None, coords=None):
        pass


def do_ground_fit():
    """
    Updates needed to MMEXOFAST:
    - Error Renormalization and result recalculation
    - Parallax grid fitting (implementation started)

    :return:
    """
    pl_fitter = ParallaxFitter(
        data_files=ground_data_files, coords=coords, fit_type='point lens')
    pl_fitter.fit()

    pl_fitter.print_latex_results_table()
    #pl_fitter.plot_lc()
    #pl_fitter.plot_parallax_grids()
    #plt.show()


def do_complete_fit():
    par_fitter = SpaceParallaxFitter(
        ground_data_files=ground_data_files, space_data_files=space_data_files, coords=coords, fit_type='point lens')
    par_fitter.fit_ground_only()
    par_fitter.fit_space_and_ground()

    par_fitter.print_latex_results_table(type='full')

    par_fitter.plot_ground_only_lc()
    par_fitter.plot_space_and_ground_lc()
    par_fitter.plot_parallax_grids(savefig='OB0939_par_contours.eps', dpi=300)


if __name__ == '__main__':
    do_ground_fit()
    #do_complete_fit()
