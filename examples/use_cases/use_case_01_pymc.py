"""
Create an interface between MM and pymc so that pymc can access event.get_chi2().

Follows: https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html

except that example is bare bones, so also adding links to additional documentation as we go.

"""
import os.path
import numpy as np
import matplotlib.pyplot as plt

import arviz as az
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import pymc as pm

import MulensModel as mm
from exozippy import MULENS_DATA_PATH


# Do we need to do separate datasets separately?
# Affects choice of mm.Event vs. mm.FitData as the input.
def ln_prob_mm(t_0, u_0, t_E, err, time, flux):
    """ combines likelihood and priors"""
    #parameters = ['t_0', 'u_0', 't_E']
    #model_parameters = {}

    #for parameter, item in zip(parameters, inputs[0:len(parameters)]):
    #    model_parameters[parameter] = item.item()
    #
    #datasets = []
    #for i in range(int((len(inputs) - len(parameters)) / 3)):
    #    datasets.append(mm.MulensData(
    #        [inputs[len(parameters) + 3 * i], inputs[len(parameters) + 3 * i + 1], inputs[len(parameters) + 3 * i + 2]],
    #        phot_fmt='flux'))

    model_parameters = {'t_0': t_0.item(), 'u_0': u_0.item(), 't_E': t_E.item()}
    datasets = [mm.MulensData([time, flux, err], phot_fmt='flux')]

    model = mm.Model(model_parameters)
    event = mm.Event(model=model, datasets=datasets)
    #event.fit_fluxes()
    #print(time[0:10])
    #print(event.get_chi2(), t_0, u_0, t_E, event.get_ref_fluxes())
    #print(event)

    chi2s = np.array(event.get_chi2_per_point()).transpose().squeeze()

    return -0.5 * chi2s


class LogLike(Op):
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
            tensor_list.append(pytensor.as_symbolic(dataset.time, name='Time_{0}'.format(i)))
            tensor_list.append(pytensor.as_symbolic(dataset.flux, name='Flux_{0}'.format(i)))
            tensor_list.append(pytensor.as_symbolic(dataset.err_flux, name='Err_{0}'.format(i)))

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
            tensor_list.append(pytensor.as_symbolic(theta[i], name=parameter))

        for parameter in model.parameters.parameters.keys():
            if parameter not in parameters_to_fit:
                tensor_list.append(
                    pt.constant(model.parameters.parameters[parameter], name=parameter))

        return tensor_list

    def make_node(self, t_0, u_0, t_E, err, time, flux) -> Apply:
        """

        :param theta:
        :param parameters_to_fit:
        :param datasets: *list* of Mulens Data objects
        :return:
        """
        # Convert inputs to tensor variables
        #parameter_tensors = self.get_parameter_tensors(event.model, theta, parameters_to_fit)
        #data_tensors = self.get_data_tensors(event.datasets)
        t_0 = pt.as_tensor(t_0)
        u_0 = pt.as_tensor(u_0)
        t_E = pt.as_tensor(t_E)
        err = pt.as_tensor(err)
        time = pt.as_tensor(time)
        flux = pt.as_tensor(flux)

        # Inputs is a list of pytensor objects.
        inputs = [t_0, u_0, t_E, err, time, flux]

        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        # outputs = [pt.vector()]
        #print([item.type() for item in data_tensors if item.name[0:4] == 'Flux'])
        # outputs = [item.type() for item in data_tensors if item.name[0:4] == 'Flux']
        outputs = [flux.type()]
        # JCY: does this mean we need the likelihood for each individual datapoint?

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        #theta, data = inputs  # this will contain my variables
        t_0, u_0, t_E, err, time, flux = inputs

        # call our numpy log-likelihood function
        loglike_eval = ln_prob_mm(t_0, u_0, t_E, err, time, flux)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)


# Import data for a simple point lens light curve
data = mm.MulensData(
    file_name=os.path.join(MULENS_DATA_PATH, 'OB140939', 'ob140939_OGLE.dat'), phot_fmt='mag')
# Initial model
model = mm.Model({'t_0': 2456836.2, 'u_0': 0.9, 't_E': 23.})
# , 'pi_E_N': -0.248, 'pi_E_E': 0.234}) # passing this extra info is hard. Also need to pass coords. ephem.
event = mm.Event(datasets=data, model=model)

parameters_to_fit = ['t_0', 'u_0', 't_E']

log_like = LogLike()

#print('\n*** Test data to pytensor ***')
#tensor_data = log_like.get_data_tensors(data)
#for item in tensor_data:
#    print(item)
#    print(item.name, item.name[0:4], item.type)
#
#print('\n*** Test model to pytensor ***')
#theta = [model.parameters.parameters[param] for param in parameters_to_fit]
#print(theta)
#tensor_params = log_like.get_parameter_tensors(model=model, theta=theta, parameters_to_fit=parameters_to_fit)
#for item in tensor_params:
#    print(item)

print('\n*** Test model + data to pytensor: ***')
node = log_like.make_node(
    model.parameters.t_0, model.parameters.u_0, model.parameters.t_E, data.err_flux, data.time, data.flux)
    #theta=theta, parameters_to_fit=parameters_to_fit, event=event)
print(node)

print('\n*** Test full pytensor class: ***')
test_out = log_like(model.parameters.t_0, model.parameters.u_0, model.parameters.t_E, data.err_flux, data.time, data.flux)
pytensor.dprint(test_out, print_type=True)

print('\n*** Test eval: ***')
test_out.eval()


print('\n*** Test actual running: ***')


def custom_dist_loglike(flux, t_0, u_0, t_E, err, date):
    # data, or observed is always passed as the first input of CustomDist
    return log_like(t_0, u_0, t_E, err, date, flux)


# use PyMC to sampler from log-likelihood
with pm.Model() as no_grad_model:
    # uniform priors on m and c
    t_0 = pm.Uniform("t_0", lower=model.parameters.t_0 - 5., upper=model.parameters.t_0 + 5., initval=model.parameters.t_0)
    u_0 = pm.Uniform("u_0", lower=0.5, upper=1.5, initval=model.parameters.u_0)
    t_E = pm.Uniform("t_E", lower=10., upper=40., initval=model.parameters.t_E)

    # use a CustomDist with a custom logp function
    likelihood = pm.CustomDist(
        "likelihood", t_0, u_0, t_E, data.err_flux, data.time, observed=data.flux, logp=custom_dist_loglike
    )

    ip = no_grad_model.initial_point()
    print('Initial point:', ip)
    no_grad_model.compile_logp(vars=[likelihood], sum=False)(ip)

    print('Expect error:')
    try:
        no_grad_model.compile_dlogp()
    except Exception as exc:
        print(type(exc))

    print('run...')
    with no_grad_model:
        # Use custom number of draws to replace the HMC based defaults
        idata_no_grad = pm.sample(5000, tune=1000, cores=1, chains=4)

    # plot the traces
    az.plot_trace(idata_no_grad, lines=[
        ("t_0", {}, model.parameters.t_0), ("u_0", {}, model.parameters.u_0), ("t_E", {}, model.parameters.t_E)])

    plt.tight_layout()
    plt.savefig('../../sandbox/trace.png', dpi=300)
    plt.show()
