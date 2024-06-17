import os.path
import exozippy as mmexo
import MulensModel

dir_ = os.path.join(mmexo.MULENS_DATA_PATH, "2018DataChallenge")

# Test data
lc_num = 4
file_w149 = os.path.join(
    dir_, 'n20180816.W149.WFIRST18.{0:03}.txt'.format(lc_num))
file_z087 = os.path.join(
    dir_, 'n20180816.Z087.WFIRST18.{0:03}.txt'.format(lc_num))
data_w149 = MulensModel.MulensData(file_name=file_w149, phot_fmt='flux')
data_z087 = MulensModel.MulensData(file_name=file_z087, phot_fmt='flux')
datasets = [data_w149, data_z087]