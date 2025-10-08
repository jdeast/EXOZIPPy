import os.path
import exozippy
import numpy as np
import pandas as pd
import MulensModel as mm
import matplotlib.pyplot as plt

dir_ = os.path.join(exozippy.MULENS_DATA_PATH, "2018DataChallenge")

event_info = np.genfromtxt(
    os.path.join(dir_, 'event_info.txt'), dtype=None, encoding='utf-8',
    names=['file', 'num', 'ra', 'dec'], usecols=range(4))


class TestDataSet():

    def __init__(self, lc_num):
        self.file_w149 = os.path.join(
            dir_, 'n20180816.W149.WFIRST18.{0:03}.txt'.format(lc_num))
        self.file_z087 = os.path.join(
            dir_, 'n20180816.Z087.WFIRST18.{0:03}.txt'.format(lc_num))

        index = np.where(event_info['num'] == lc_num)
        self.coords = '{0} {1}'.format(
            event_info['ra'][index][0], event_info['dec'][index][0])


class DC18Answers():

    def __init__(self):
        columns = np.genfromtxt(
            os.path.join(dir_, 'Answers', 'wfirstColumnNumbers.txt'), dtype=None, encoding='utf-8',
            usecols=[0, 1], skip_header=2, names=['index', 'name'])
        index = np.where(columns['name'] == '|')
        for i in index[0]:
            columns[i]['name'] = 'col{0}'.format(i)

        self.names = [name for name in columns['name']]
        self.filename = os.path.join(dir_, 'Answers', 'master_file.txt')
        self.data = pd.read_csv(
            self.filename,
            names=self.names, usecols=range(len(self.names)),
            sep='\s+', skiprows=1,
        )
        self.classes = self._read_classes()
        #print('The columns may NOT be read in correctly!')

    def _read_classes(self):
        classes = []
        with open(self.filename, 'r') as orig_file:
            for line in orig_file.readlines()[1:]:
                label = line.split(' ')[-2]
                class_type = label.split('_')[0]
                classes.append(class_type)

        return pd.Series(classes)

    def check_table(self):
        orig_file = open(self.filename, 'r')
        lines = orig_file.readlines()
        for i, line in enumerate(lines[1:]):
            #print(i, line[-20:])
            sections = line.split('|')
            ulens_1 = sections[3].split()
            u0 = float(ulens_1[0])
            alpha = float(ulens_1[1])
            t0 = float(ulens_1[2])
            tE = float(ulens_1[3])
            rhos = float(ulens_1[7])

            ulens_2 = sections[4].split()
            q = float(ulens_2[4])
            s = float(ulens_2[5])
            if (q < 0.01) and ('cassan' in line):
                print(i, line)
                print(ulens_1)
                print(ulens_2)
                print(self.data.iloc[i][['idx', 'u0', 'alpha', 't0', 'tE', 'rhos', 'q', 's']])

            #if (
            #        (t0 != self.data['t0'].iloc[i]) or
            #        (u0 != self.data['u0'].iloc[i]) or
            #        (tE != self.data['tE'].iloc[i]) or
            #        (rhos != self.data['rhos'].iloc[i]) or
            #        (s != self.data['s'].iloc[i]) or
            #        (q != self.data['q'].iloc[i]) or
            #        (alpha != self.data['alpha'].iloc[i])
            #):
            #    print(i, 'Read Error')
            #    print(sections[3])
            #    print(sections[4])
            #    print('t0', t0, self.data['t0'].iloc[i])
            #    print('u0', u0, self.data['u0'].iloc[i])
            #    print('tE', tE, self.data['tE'].iloc[i])
            #    print('rhos', rhos, self.data['rhos'].iloc[i])
            #    print('s', s, self.data['s'].iloc[i])
            #    print('q', q, self.data['q'].iloc[i])
            #    print('alpha', alpha, self.data['alpha'].iloc[i])

        orig_file.close()

    def print_wide_orbit_planets(self):
        u0_index = self.data['u0'] > 0.1
        s_index = self.data['s'] > 1.
        q_index = self.data['q'] < 0.03
        with pd.option_context('display.width', None, 'display.max_rows', None):
            print(
                self.data[u0_index & s_index & q_index][
                    ['idx', 'ra', 'dec', 't0', 'u0', 'tE', 'rhos', 'piE', 's', 'q', 'alpha']].sort_values('idx'))


    def plot_model(self, num):
        index = self.data.index == (num - 1)
        print(np.sum(index))
        print(self.data[index])
        values = self.data[index].iloc[0]
        params = {
            't_0': values['t0'], 'u_0': values['u0'], 't_E': values['tE'], 'rho': values['rhos'],
            's': values['s'], 'q': values['q'], 'alpha': values['alpha'],
        }
        model = mm.Model(params)
        print(model)
        plt.figure()
        model.plot_magnification()
        plt.figure()
        model.plot_trajectory(caustics=True)
        plt.show()

if __name__ == '__main__':
    answers = DC18Answers()
    event_info = pd.DataFrame(event_info, index=event_info['num'])
    answers.data = answers.data.merge(event_info)
    answers.print_wide_orbit_planets()
    #print(answers.data.columns)
    print(answers.data[['idx', 'subrun', 'field', 'src_id', 'lens_id', 'file_', 'num']])
    answers.plot_model(131)
