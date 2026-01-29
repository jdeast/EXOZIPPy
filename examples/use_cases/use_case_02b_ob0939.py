"""
Analyze the microlensing event OB140939 and produce all outputs for the paper.
"""
import exozippy
import numpy as np
import MulensModel as mm
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
        exozippy.MULENS_DATA_PATH, 'OB140939', 'n20140605.L.Spitzer.OB140939.txt')]
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
        print(self.fitter.make_ulens_table('latex'))

    def save_results_table(self, filename=None, type=None):
        if filename is None:
            raise KeyError('You must provide a filename.')

        with open(filename, 'w') as file_:
            file_.write(self.fitter.make_ulens_table('latex'))

    def plot_lc(self, savefig=None, **kwargs):
        # These lines suggest the need for some method in self.fitter:
        # self.fitter.get_model('static PSPL')
        model = self.fitter.all_fit_results.get('PSPL static ').full_result.fitter.get_model()
        model.parameters = self.fitter.all_fit_results.get('PSPL static ').params
        event = mm.Event(datasets=self.fitter.datasets, model=model)
        event.plot()

        if savefig is not None:
            plt.savefig(savefig, **kwargs)

    def plot_parallax_grids(self, n_scale=1, savefig=None, **kwargs):
        ncols = 2
        fig, axes = plt.subplots(1, ncols, figsize=(2.25*ncols, 5))
        for idx, u0sign in enumerate(['+', '-']):
            grid = self.fitter.results['PSPL par ({0}u_0)'.format(u0sign)].grid
            best_chi2 = self.fitter.results['PSPL par ({0}u_0)'.format(u0sign)].best['chi2']
            plt.sca(axes[idx])
            plt.scatter(
                grid['pi_E_E'], grid['pi_E_N'], c=np.sqrt(grid['chi2'] - best_chi2),
                vmin=0, vmax=n_scale*9, cmap='Set1')
            plt.colorbar(label=r'$\sigma$')

            axes[idx].tick_params(axis='both', which='both', direction='in',
                       top=True, right=True, bottom=True, left=True)
            axes[idx].invert_xaxis()
            axes[idx].set_xlabel(r'$\pi_{\rm E,E}$')
            axes[idx].set_aspect('equal')
            axes[idx].set_title("{0}u0".format(u0sign))

            if idx == 0:
                axes[idx].set_ylabel(r'$\pi_{\rm E,N}$')
            else:
                axes[idx].tick_params(axis='y', which='both', labelleft=False)
                axes[idx].set_ylabel('')

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(savefig, **kwargs)


    @property
    def fitter(self):
        return self._fitter

    @fitter.setter
    def fitter(self, value):
        self._fitter = value


class SpaceParallaxFitter(ParallaxFitter):

    def __init__(self, ground_data_files=None, space_data_files=None, **kwargs):
        super(SpaceParallaxFitter, self).__init__(**kwargs)
        self.ground_data_files = ground_data_files
        self.space_data_files = space_data_files

    def do_fits(self):
        pass


def do_ground_fit():
    """
    Updates needed to MMEXOFAST:
    - Error Renormalization and result recalculation
    - Parallax grid fitting (implementation started)
    - Handling of zero blending cases.

    :return:
    """
    pl_fitter = ParallaxFitter(
        data_files=ground_data_files, coords=coords, fit_type='point lens')
    pl_fitter.fit(verbose=True)

    print(pl_fitter.fitter.all_fit_results)
    print(pl_fitter.fitter.make_ulens_table('ascii'))
    pl_fitter.save_results_table(filename='test_output/OB0939_gr_fits.tex', type='latex')
    pl_fitter.plot_lc(savefig='test_output/OB0939_gr_lc.eps', dpi=300)
    #pl_fitter.plot_parallax_grids(savefig='test_output/OB0939_gr_piEgrid.eps', dpi=300)
    plt.show()


def do_complete_fit():
    par_fitter = SpaceParallaxFitter(
        ground_data_files=ground_data_files, space_data_files=space_data_files, coords=coords, fit_type='point lens')
    par_fitter.do_fits()
    par_fitter.save_results_table(filename='test_output/OB0939_fits.tex', type='latex')

    par_fitter.plot_space_and_ground_lc(savefig='test_output/OB0939_lc.eps', dpi=300)
    par_fitter.plot_parallax_contours(savefig='test_output/OB0939_par_contours.eps', dpi=300)


if __name__ == '__main__':
    do_ground_fit()
    #do_complete_fit()
