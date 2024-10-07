from os import path
from exozippy.mmexofast import gridsearches, mmexofast, ulens, estimate_params

MODULE_PATH = path.abspath(path.join(path.dirname(__file__), '..'))
DATA_PATH = path.join(MODULE_PATH, 'data')
MULENS_DATA_PATH = path.join(DATA_PATH, 'mulens')
