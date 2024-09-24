import os.path, glob
import numpy as np
import matplotlib.pyplot as plt

from examples.DC18_classes import DC18Answers
from plot_results import PlanetFitInfo


class AllResults():
    pspl_fit_types = ['Initial Guess', 'Initial SFIT', 'Revised SFIT']

    def __init__(self, path='.'):
        self.results = self.get_results(path)
        self.answers = self.get_answers()

        self._delta_t_0 = None
        self._delta_u_0 = None
        self._delta_t_E = None

    def get_results(self, path):
        results = {}
        logs = glob.glob(os.path.join('temp_output', '4u0values', 'WFIRST*.log'))
        for file in np.sort(logs):
            planet = PlanetFitInfo(file)
            results[planet.lc_num] = planet

        return results

    def get_answers(self):
        all_answers = DC18Answers()
        answers = []
        # Probably needs to be rewritten to concatenate Pandas DF.
        for key in self.results.keys():
            answers.append(all_answers.get_ulens_params(key))

        return answers

    def plot_delta_t_0(self):
        plt.figure()
        for fit_type in AllResults.pspl_fit_types:
            plt.hist(self.delta_t_0[fit_type], label=fit_type, bins=20)

        plt.xlabel(r'$\Delta t_0')
        plt.minorticks_on()

    def plot_delta_u_0(self, frac=True):
        plt.figure()
        for fit_type in AllResults.pspl_fit_types:
            if frac:
                x = self.delta_u_0[fit_type] / self.answers['u_0']
            else:
                x = self.plot_delta_u_0()

            plt.hist(x, label=fit_type, bins=20)

        if frac:
            plt.xlabel(r'$\Delta u_0 / u_0')
        else:
            plt.xlabel(r'$\Delta u_0')

        plt.minorticks_on()

    def plot_delta_t_E(self, frac=True):
        plt.figure()
        for fit_type in AllResults.pspl_fit_types:
            if frac:
                x = self.delta_t_E[fit_type] / self.answers['t_E']
            else:
                x = self.plot_delta_t_E()

            plt.hist(x, label=fit_type, bins=20)

        if frac:
            plt.xlabel(r'$\Delta t_E / t_E')
        else:
            plt.xlabel(r'$\Delta t_E')

        plt.minorticks_on()

    @property
    def delta_t_0(self):
        raise NotImplementedError()
        if self._delta_t_0 is None:
            delta_t_0 = np.ones((len(self.results.keys()), len(AllResults.pspl_fit_types)))
            for fit_type in AllResults.pspl_fit_types:
                if fit_type == 'Initial Guess':
                    pass
                elif fit_type == 'Initial SFIT':
                    pass
                elif fit_type == 'Revised SFIT':
                    pass

        return self._delta_t_0

    @property
    def delta_u_0(self):
        raise NotImplementedError()
        return self._delta_u_0

    @property
    def delta_t_E(self):
        raise NotImplementedError()
        return self._delta_t_E


if __name__ == '__main__':
    results = AllResults(path=os.path.join('temp_output', '4u0values'))
    results.plot_delta_t_0()
    results.plot_delta_u_0()
    results.plot_delta_t_E()