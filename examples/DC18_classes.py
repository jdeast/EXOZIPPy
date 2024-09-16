import os.path
import exozippy
import numpy as np
import pandas as pd

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
        self.data = pd.read_csv(
            os.path.join(dir_, 'Answers', 'master_file.txt'),
            names=self.names, usecols=range(len(self.names)),
            delim_whitespace=True, skiprows=1
        )


if __name__ == '__main__':
    answers = DC18Answers()
    print(answers.data)
    print(event_info.shape)