import numpy as np


class AnomalyClassifier(object):

    def __init__(self):
        pass

    def classify(self, params):
        """
        Use the lightcurve and anomaly properties to determine what kind of fit is needed.

        :param params: results of AnomalyPropertyEstimator.get_anomaly_lc_parameters()
        :return: *str*
            one of 'close', 'wide', 'high_mag'
        """
        if np.abs(params['u_0']) < 0.05:
            return 'high_mag'

        if params['dmag'] < 0:
            return 'wide'

        if params['dmag'] > 0:
            return 'close'
