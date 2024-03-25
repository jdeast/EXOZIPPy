import os.path
import numpy as np
import astropy.units as u
import sys
import yaml

import MulensModel as mm
import mmexofast as mmexo

from print_ex_metrics import print_metrics


def divide_settings(settings):
	"""
	divide settings into expected, files, and all others
	"""
	out = {**settings}
	expected = out.pop('expected')
	files_settings = out.pop('files')

	return (out, expected, files_settings)


def process_files(files, root=None, subdir=None):
	"""
	make a list with list of files
	"""
	if root is None:
		root_path = ""
	elif root == "MODULE_PATH":
		root_path = mm.MODULE_PATH
	else:
		raise ValueError("root not recognized: " + str(root))

	if subdir is not None:
		root_path = os.path.join(root_path, subdir)

	if not isinstance(files, list):
		raise TypeError('wrong value of files kwarg: ' + str(type(files)))

	out = []
	for file_ in files:
 		out.append(os.path.join(root_path, file_))

	return out


if __name__ == '__main__':
	if len(sys.argv) != 2:
		raise ValueError('single yaml file with settings is needed')

	with open(sys.argv[1]) as in_file:
		settings = yaml.safe_load(in_file)

	(kwargs, expected, file_settings) = divide_settings(settings)

	files = process_files(**file_settings)
	kwargs['files'] = files

	# u.M_e has to be parsed.

	print(kwargs)
	print()
	print(expected)

	results = mmexo.mmexofast.fit(**kwargs)

	print_metrics(results.final_parameters, expected)
