"""
Create an interface between MM and pymc so that pymc can access event.get_chi2().

Follows: https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html

except that example is bare bones, so also adding links to additional documentation as we go.

"""
import os.path
import numpy as np

import pytensor as pt
import pymc as pm

import MulensModel as mm
from exozippy import MULENS_DATA_PATH


# Do we need to do separate datasets separately?
# Affects choice of mm.Event vs. mm.FitData as the input.
def ln_prob_mm(inputs):
    """ combines likelihood and priors"""

    model_parameters = {}
    data_objects = {}
    for item in inputs:
        if (item.name[0:3] == 'Tim') or (item.name[0:3] == 'Flu') or (item.name[0:3] == 'Err'):
            name_elements = item.name.split('_')
            key = name_elements[0]
            num = name_elements[-1]
            if num in data_objects.keys():
                data_objects[num][key] = item

        else:
            model_parameters[item.name] = item

    datasets = []
    data_keys = ['Time', 'Flux', 'Err']
    for value in data_objects.values():
        datasets.append(mm.MulensData(
            [value[key] for key in data_keys], phot_fmt='flux'
        ))

    print(model_parameters)
    model = mm.Model(model_parameters)
    event = mm.Event(model=model, datasets=datasets)

    chi2s = event.get_chi2_per_point()

    return -0.5 * chi2s


class LogLike(pt.graph.Op):
    """
    Op documentation: https://pytensor.readthedocs.io/en/latest/extending/op.html
    """

    def get_data_tensors(self, datasets):
        """
        pymc Data containers:
        https://www.pymc.io/projects/examples/en/latest/fundamentals/data_container.html

        See example about WHO data on baby length.

        Input:
            datasets = *list* of *MulensModel.MulensData* objects

        Output:
            tensor_list = *list* with each component of each dataset converted to a pymc.Data object.

            JCY: not sure if this is okay or if the data need to be converted to pytensors, specifically. i.e.,
            Is pt.as_tensor() different from pymc.Data() in a meaningful way?

        """
        if isinstance(datasets, mm.MulensData):
            datasets = [datasets]
        elif not isinstance(datasets, list):
            raise TypeError('datasets must be a *list* not', type(datasets))

        tensor_list = []
        for i, dataset in enumerate(datasets):
            if not isinstance(dataset, mm.MulensData):
                raise TypeError(
                    'dataset {0} must be MulensModel.MulensData object, not {1}'.format(i,  type(dataset)))

            # pm.Data(name, vector) didn't work
            tensor_list.append(pt.as_symbolic(dataset.time, name='Time_{0}'.format(i)))
            tensor_list.append(pt.as_symbolic(dataset.flux, name='Flux_{0}'.format(i)))
            tensor_list.append(pt.as_symbolic(dataset.err_flux, name='Err_{0}'.format(i)))

        return tensor_list

    def get_parameter_tensors(self, model, theta, parameters_to_fit):
        """
        https://pytensor.readthedocs.io/en/latest/library/index.html
        JCY: I'm thinking that as_symbolic replaced as_tensor at some point.

        :param theta:
        :param parameters_to_fit:
        :return:
        """

        tensor_list = []
        for i, parameter in enumerate(parameters_to_fit):
            tensor_list.append(pt.as_symbolic(theta[i], name=parameter))

        for parameter in model.parameters.parameters.keys():
            if parameter not in parameters_to_fit:
                tensor_list.append(
                    pt.as_symbolic(model.parameters.parameters[parameter], name=parameter))

        return tensor_list

    def make_node(self, theta, parameters_to_fit, event) -> pt.graph.Apply:
        """

        :param theta:
        :param parameters_to_fit:
        :param datasets: *list* of Mulens Data objects
        :return:
        """
        # Convert inputs to tensor variables
        parameter_tensors = self.get_parameter_tensors(event.model, theta, parameters_to_fit)
        data_tensors = self.get_data_tensors(event.datasets)

        # Inputs is a list of pytensor objects.
        inputs = parameter_tensors + data_tensors

        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        # outputs = [pt.vector()]
        #print([item.type() for item in data_tensors if item.name[0:4] == 'Flux'])
        outputs = [item.type() for item in data_tensors if item.name[0:4] == 'Flux']
        # JCY: does this mean we need the likelihood for each individual datapoint?

        # Apply is an object that combines inputs, outputs and an Op (self)
        return pt.graph.Apply(self, inputs, outputs)

    def perform(self, node: pt.graph.Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        #theta, data = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = ln_prob_mm(inputs)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)


# Import data for a simple point lens light curve
data = mm.MulensData(
    file_name=os.path.join(MULENS_DATA_PATH, 'OB140939', 'ob140939_OGLE.dat'), phot_fmt='mag')
# Initial model
model = mm.Model({'t_0': 6836.3, 'u_0': 0.9, 't_E': 23., 'pi_E_N': -0.248, 'pi_E_E': 0.234})
event = mm.Event(datasets=data, model=model)

parameters_to_fit = ['t_0', 'u_0', 't_E']

log_like = LogLike()

print('*** Test data to pytensor ***')
tensor_data = log_like.get_data_tensors(data)
for item in tensor_data:
    print(item)
    print(item.name, item.name[0:4], item.type)

print('*** Test model to pytensor ***')
theta = [model.parameters.parameters[param] for param in parameters_to_fit]
print(theta)
tensor_params = log_like.get_parameter_tensors(model=model, theta=theta, parameters_to_fit=parameters_to_fit)
for item in tensor_params:
    print(item)

print('*** Test model + data to pytensor: ***')
node = log_like.make_node(theta=theta, parameters_to_fit=parameters_to_fit, event=event)
print(node)

print('*** Test full pytensor class: ***')
test_out = log_like(theta, parameters_to_fit, event)
print(test_out)





