import MulensModel
import matplotlib.pyplot as plt
import numpy as np
import glob, os.path
import exozippy as mmexo
from examples.use_cases.DC18_classes import TestDataSet


class PlanetFitInfo():

    def __init__(self, filename):
        with open(filename, 'r') as file_:
            lines = file_.readlines()

        self.lines = lines
        self.lc_num = int(os.path.basename(filename).split('.')[1])

        self._data = None
        self._fitter = None
        self._initial_pspl_guess = None
        self._sfit_params = None
        self._revised_sfit_params = None
        self._best_af_grid_point = None
        self._initial_planet_params = None
        self._mag_methods = None

    def extract_params(self, line):
        """Extract key-value pairs from a dict literal embedded in a log line.

        Locates the opening '{' automatically, so this works for both the old
        plain-text format and the new timestamp-prefixed logging format.
        """
        elements = line.split()
        params = {}

        # Find the element that opens the dict literal
        start_idx = None
        for i, elem in enumerate(elements):
            if elem.startswith('{'):
                start_idx = i
                break

        if start_idx is None:
            return params

        for i in range(start_idx, len(elements), 2):
            key = elements[i].strip("{:'\"")
            if not key:
                continue
            if i + 1 >= len(elements):
                break
            if '<' in elements[i + 1]:
                if i + 2 < len(elements):
                    value = float(elements[i + 2].strip(',}'))
                else:
                    continue
            elif '>' in elements[i + 1]:
                continue
            else:
                try:
                    value = float(elements[i + 1].strip(',}'))
                    params[key] = value
                except ValueError:
                    pass

        return params

    @property
    def mag_methods(self):
        if self._mag_methods is None:
            for line in self.lines:
                # Handle both "mag methods" (new) and "mag_methods" (old)
                if 'mag methods' in line or 'mag_methods' in line:
                    bracket_pos = line.find('[')
                    if bracket_pos == -1:
                        continue
                    mag_methods = []
                    for element in line[bracket_pos:].split():
                        cleaned = element.strip("[],'\"")
                        if not cleaned:
                            continue
                        try:
                            mag_methods.append(float(cleaned))
                        except ValueError:
                            mag_methods.append(cleaned)

                    self._mag_methods = mag_methods

        return self._mag_methods

    @property
    def initial_pspl_guess(self):
        if self._initial_pspl_guess is None:
            for line in self.lines:
                # Matches both "Initial PSPL" (old) and "Initial PSPL Estimate" (new)
                if 'Initial PSPL' in line:
                    self._initial_pspl_guess = self.extract_params(line)

        return self._initial_pspl_guess

    @property
    def sfit_params(self):
        if self._sfit_params is None:
            for line in self.lines:
                # "SFIT params" (old) -> "Initial SFit" (new)
                if 'Initial SFit' in line:
                    params = self.extract_params(line)
                    params.pop('chi2', None)  # chi2 is now included but isn't a model param
                    self._sfit_params = params

        return self._sfit_params

    @property
    def revised_sfit_params(self):
        if self._revised_sfit_params is None:
            for line in self.lines:
                if 'Revised SFIT' in line:
                    self._revised_sfit_params = self.extract_params(line)

        return self._revised_sfit_params

    @property
    def initial_planet_params(self):
        if self._initial_planet_params is None:
            for line in self.lines:
                # Matches both "Initial 2L1S params" (old) and "Initial 2L1S Wide Model" (new)
                if 'Initial 2L1S' in line:
                    self._initial_planet_params = self.extract_params(line)

        return self._initial_planet_params

    @property
    def best_af_grid_point(self):
        if self._best_af_grid_point is None:
            for line in self.lines:
                if 'Best AF' in line:
                    self._best_af_grid_point = self.extract_params(line)

        return self._best_af_grid_point

    def get_event_t_range(self, n_tE=3):
        params = None
        if self.initial_planet_params is not None:
            params = self.initial_planet_params
        elif self.revised_sfit_params is not None:
            params = self.revised_sfit_params
        elif self.sfit_params is not None:
            params = self.sfit_params

        start = params['t_0'] - n_tE * params['t_E']
        stop =  params['t_0'] + n_tE * params['t_E']
        return [start, stop]

    @property
    def planet_t_range(self):
        if self.mag_methods is not None:
            return [self.mag_methods[0], self.mag_methods[-1]]
        elif self.best_af_grid_point is not None:
            n_teff = 3
            start = self.best_af_grid_point['t_0'] - n_teff * self.best_af_grid_point['t_eff']
            stop = self.best_af_grid_point['t_0'] + n_teff * self.best_af_grid_point['t_eff']
            return [start, stop]
        else:
            return self.get_event_t_range(n_tE=1)

    @property
    def data(self):
        if self._data is None:
            self._data = self.fitter.datasets

        return self._data

    @property
    def fitter(self):
        if self._fitter is None:
            data = TestDataSet(self.lc_num)
            self._fitter = mmexo.mmexofast.MMEXOFASTFitter(files=[data.file_w149, data.file_z087])
            if self.sfit_params is not None:
                self._fitter.pspl_params = self.sfit_params

            if self.best_af_grid_point is not None:
                self._fitter.best_af_grid_point = self.best_af_grid_point
                #self._fitter.set_datasets_with_anomaly_masked(mask_type='t_eff')

        return self._fitter

    def plot_planet_window(self):
        if self.best_af_grid_point is not None:
            plt.axvline(self.best_af_grid_point['t_0'] - 2450000., color='black', linestyle=':')
            plt.axvline(
                self.best_af_grid_point['t_0'] - self.best_af_grid_point['t_eff'] - 2450000.,
                color='black', linestyle='--')
            plt.axvline(
                self.best_af_grid_point['t_0'] + self.best_af_grid_point['t_eff'] - 2450000.,
                color='black', linestyle='--')

    def make_plot(self, event, n_tE=5):
        plt.figure(figsize=(10, 6))
        plt.suptitle('({0}): {1}'.format(self.lc_num, event.model))
        plt.subplot(1, 2, 1)
        event.plot_data(show_bad=True, subtract_2450000=True)
        event.plot_model(
            t_range=self.get_event_t_range(n_tE=n_tE),
            subtract_2450000=True, color='black', zorder=10)
        self.plot_planet_window()
        plt.xlim(np.array(self.get_event_t_range(n_tE=n_tE)) - 2450000.)
        plt.minorticks_on()

        plt.subplot(1, 2, 2)
        event.plot_data(show_bad=True, subtract_2450000=True)
        event.plot_model(t_range=self.planet_t_range, color='black', subtract_2450000=True, zorder=10)
        self.plot_planet_window()
        plt.xlim(np.array(self.planet_t_range) - 2450000.)
        plt.minorticks_on()

        plt.tight_layout()

    def plot_initial_pspl_fit(self):
        model = MulensModel.Model(self.sfit_params)
        event = MulensModel.Event(datasets=self.data, model=model)
        self.make_plot(event)

    def plot_revised_pspl_fit(self):
        model = MulensModel.Model(self.revised_sfit_params)
        event = MulensModel.Event(datasets=self.fitter.masked_datasets, model=model)
        self.make_plot(event)

    def plot_initial_planet_model(self, trajectory=False):
        model = MulensModel.Model(self.initial_planet_params)
        model.set_magnification_methods(self.mag_methods)
        event = MulensModel.Event(datasets=self.fitter.masked_datasets, model=model)
        self.make_plot(event)
        if trajectory:
            plt.figure()
            model.plot_trajectory(caustics=True)
            plt.gca().set_aspect('equal')
            plt.gca().minorticks_on()


if __name__ == '__main__':
    import os.path

    logs = glob.glob(os.path.join('temp_output', 'WFIRST.*.log'))
    for file in np.sort(logs):
        print(os.path.basename(file))
        planet = PlanetFitInfo(file)
        if planet.sfit_params is not None:
            planet.plot_initial_pspl_fit()
        else:
            print('SFIT Failed!')

        if planet.revised_sfit_params is not None:
            planet.plot_revised_pspl_fit()
        else:
            print('No revised SFIT model!')
            print('AF Results: ', planet.best_af_grid_point)

        if planet.initial_planet_params is not None:
            try:
                planet.plot_initial_planet_model(trajectory=True)
            except:
                print('Planet plotting failed.')

        else:
            print('No 2L1S model!')

        plt.show()
