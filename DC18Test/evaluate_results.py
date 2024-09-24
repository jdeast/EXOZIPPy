import os.path, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from examples.DC18_classes import DC18Answers
from plot_results import PlanetFitInfo


class AllResults():
    pspl_fit_types = ['Initial PSPL Guess', 'Initial SFIT', 'Revised SFIT', '2L1S Guess']

    def __init__(self, path='.'):
        self.results = self.get_results(path)
        self.answers = self.get_answers()

        self._delta_t_0 = None
        self._delta_u_0 = None
        self._delta_t_E = None

    def get_results(self, path):
        results = {}
        for key in AllResults.pspl_fit_types:
            results[key] = None

        logs = glob.glob(os.path.join(path, 'WFIRST*.log'))
        for file in np.sort(logs):
            planet = PlanetFitInfo(file)
            for fit_type, params in zip(
                    AllResults.pspl_fit_types,
                    [planet.initial_pspl_guess, planet.sfit_params, planet.revised_sfit_params,
                     planet.initial_planet_params]):

                print(fit_type, params)

                df = {'ID': planet.lc_num}
                print({**df, **params})
                if params is not None:
                    df = pd.Series(data={**df, **params})
                else:
                    df = pd.Series(df)

                print(df)

                if results[fit_type] is None:
                    results[fit_type] = df
                else:
                    results[fit_type] = pd.concat((results[fit_type], df))

                print(results[fit_type].iloc[-1])

        return results

    def get_answers(self):
        all_answers = DC18Answers()
        answers = None
        for key in self.results['Initial PSPL Guess']['ID'].values:

            df = pd.concat((pd.Series(data={'ID': key}), all_answers.data.iloc(key - 1)))
            if answers is None:
                answers = df
            else:
                answers = pd.concat((answers, df))

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
                x = self.delta_u_0[fit_type] / self.answers['u0']
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
                x = self.delta_t_E[fit_type] / self.answers['tE']
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
        if self._delta_t_0 is None:
            delta_t_0 = {}
            for fit_type in AllResults.pspl_fit_types:
                delta_t_0[fit_type] = self.answers['t0'] - self.results[fit_type]['t_0']

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
    #results.plot_delta_u_0()
    #results.plot_delta_t_E()
