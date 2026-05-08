import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.gradient import DisconnectedType
import numpy as np
import MulensModel as mm
from astropy.coordinates import SkyCoord
import astropy.units as u


def setup_mulens_model(p, coords, mag_method):
    # p = [t_0, u_0, t_E, pi_E_N, pi_E_E, (optional) rho]
    mm_params = {
        't_0': p[0], 'u_0': p[1], 't_E': p[2],
        'pi_E_N': p[3], 'pi_E_E': p[4]
    }

    t_0 = p[0]
    u_0 = np.sign(p[1]) * max(abs(p[1]), 1e-9)
    t_E = max(p[2], 1e-4)
    mm_params = {
        't_0': t_0, 'u_0': u_0, 't_E': t_E,
        'pi_E_N': p[3], 'pi_E_E': p[4]
    }

    rho_val = None
    if len(p) > 5:
        rho_val = p[5]
        mm_params['rho'] = rho_val

    model = mm.Model(parameters=mm_params, coords=coords)
    is_binary = 'q' in mm_params or 's' in mm_params

    # Logic Switchboard
    if mag_method == "auto_vbbl":
        if rho_val is not None:
            # Determine method based on geometry
            fs_method = 'VBM' if is_binary else 'finite_source_LD_Lee09'

            t_0, t_E = mm_params['t_0'], mm_params['t_E']
            window = 5.0 * rho_val * t_E

            # This is the correct list syntax: [time, method, time]
            # It means: use point_source until t_0-window, then fs_method,
            # then point_source again after t_0+window.
            model.set_magnification_methods([
                t_0 - window, fs_method, t_0 + window
            ])
        else:
            # Use 0.0 as a 'start of time' epoch to satisfy the TypeError
            model.set_magnification_methods([0.0, 'point_source'])
    else:
        # If user passed a specific string, use it for all time
        model.set_magnification_methods([0.0, mag_method])

    return model

class MulensMagOp(Op):
    """
    A PyTensor Op that wraps MulensModel.Model.get_magnification()
    Inputs: [params, times, obs_pos]
    """
    itypes = [pt.dvector, pt.dvector, pt.dmatrix]
    otypes = [pt.dvector]

    def __init__(self, coords, mag_method="point_source"):
        self.coords = coords
        self._coord_cache = {}
        self.mag_method = mag_method

    def infer_shape(self, fgraph, node, input_shapes):
        # The output is a vector the same length as the 'times' input
        return [input_shapes[1]]

    def _get_sat_coord(self, obs_pos_np):
        # Force 2D in case PyMC evaluates a single slice
        obs_pos_2d = np.atleast_2d(obs_pos_np)
        n_points = len(obs_pos_2d)

        # Build it once per array shape, then cache it
        if n_points not in self._coord_cache:
            self._coord_cache[n_points] = SkyCoord(
                x=obs_pos_2d[:, 0] * u.au,
                y=obs_pos_2d[:, 1] * u.au,
                z=obs_pos_2d[:, 2] * u.au,
                representation_type='cartesian'
            )
        return self._coord_cache[n_points]

    def perform(self, node, inputs, outputs):
        p, times_np, obs_pos_np = inputs

        model = setup_mulens_model(p, self.coords, self.mag_method)

        sat_coord = self._get_sat_coord(obs_pos_np)
        mags = model.get_magnification(times_np, satellite_skycoord=sat_coord)
        outputs[0][0] = np.asarray(mags)
        return

    def grad(self, inputs, output_grads):
        """
        Calculates the first derivative.
        inputs: [p, times, obs_pos] (3 values)
        """
        p, times, obs_pos = inputs  # ONLY unpack 3 here!
        g = output_grads[0]  # This is the 4th value for the next Op

        # We call the GradOp which takes all 4 values
        return [
            MulensMagGradOp(self.coords, self.mag_method)(p, times, obs_pos, g),
            DisconnectedType()(),
            DisconnectedType()()
        ]

    def connection_pattern(self, node):
        return [
            [True],  # wrt p
            [False],  # wrt times
            [False]  # wrt obs_pos
        ]


class MulensMagGradOp(Op):
    """Numerical gradient for MulensMagOp"""
    itypes = [pt.dvector, pt.dvector, pt.dmatrix, pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, coords, mag_method="point_source", eps=1e-6):
        self.coords = coords
        self.mag_method = mag_method
        self.eps = eps
        self._coord_cache = {}

    def infer_shape(self, fgraph, node, input_shapes):
        # The output of the grad Op is the gradient wrt parameters
        # so it must be the same shape as the parameter vector 'p'
        return [input_shapes[0]]

    def _get_sat_coord(self, obs_pos_np):
        obs_pos_2d = np.atleast_2d(obs_pos_np)
        n_points = len(obs_pos_2d)

        if n_points not in self._coord_cache:
            self._coord_cache[n_points] = SkyCoord(
                x=obs_pos_2d[:, 0] * u.au,
                y=obs_pos_2d[:, 1] * u.au,
                z=obs_pos_2d[:, 2] * u.au,
                representation_type='cartesian'
            )
        return self._coord_cache[n_points]

    def perform(self, node, inputs, outputs):
        params, times_np, obs_pos_np, g = inputs
        out = np.zeros(params.shape, dtype=params.dtype)

        times_1d = np.atleast_1d(times_np)
        sat_coord = self._get_sat_coord(obs_pos_np)

        f_x = self._calc(params, times_1d, sat_coord)

        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += self.eps
            f_plus = self._calc(p_plus, times_1d, sat_coord)

            diff = (f_plus - f_x) / self.eps
            # np.sum handles the dot product safely regardless of 1D vs 0D shapes
            out[i] = np.sum(g * diff)

        outputs[0][0] = out

    def _calc(self, p, times_1d, sat_coord):
        model = setup_mulens_model(p, self.coords, self.mag_method)
        return model.get_magnification(times_1d, satellite_skycoord=sat_coord)

    def grad(self, inputs, output_grads):
        """
        Check_curvatures is asking for a second derivative (Hessian).
        Numerical second derivatives are too noisy and expensive.
        We explicitly disconnect this from the graph to bypass the NotImplementedError.
        """

        # we don't support 2nd derivatives
        return [
            DisconnectedType()(),  # d2A/dp2 - Disconnect to solve shape mismatch
            DisconnectedType()(),  # d2A/dtimes
            DisconnectedType()(),  # d2A/dobs
            output_grads[0]  # d2A/dg - Identity for the chain rule
        ]

        # We have 4 inputs: [params, times, obs_pos, g]
        # Returning DisconnectedType()() tells PyTensor to treat the 2nd derivative as zero/un-trackable.
        p, times, obs_pos, g = inputs
        return [
            pt.zeros_like(p),  # d/dp
            DisconnectedType()(),  # d/dtimes
            DisconnectedType()(),  # d/dobs
            output_grads[0]  # d/dg (Pass through the gradient for the chain rule)
        ]

    def connection_pattern(self, node):
        """
        Tells PyTensor which inputs have a path to the output.
        Inputs are: [p, times, obs_pos, grad_output]
        """
        return [
            [True],  # Output depends on parameters (p)
            [False],  # Output does not depend on times
            [False],  # Output does not depend on obs_pos
            [True]  # Output DEFINITELY depends on the incoming grad_output
        ]
