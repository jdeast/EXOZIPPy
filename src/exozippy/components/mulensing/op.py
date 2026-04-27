import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import numpy as np
import MulensModel as mm
from astropy.coordinates import SkyCoord
import astropy.units as u

class MulensMagOp(Op):
    """
    A PyTensor Op that wraps MulensModel.Model.get_magnification()
    """
    itypes = [pt.dvector]  # [t0, u0, tE, pi_E_N, pi_E_E, rho]
    otypes = [pt.dvector]

    def __init__(self, times, coords, obs_pos):
        self.times = times
        self.coords = coords
        self.obs_skycoord = SkyCoord(
            x=obs_pos[:, 0] * u.au,
            y=obs_pos[:, 1] * u.au,
            z=obs_pos[:, 2] * u.au,
            representation_type='cartesian'
        )

    def perform(self, node, inputs, outputs):
        p = inputs[0]

        # Unpack the standard vector
        # (Order must be strictly maintained between component and op)
        mm_params = {
            't_0': p[0],
            'u_0': p[1],
            't_E': p[2],
            'pi_E_N': p[3],
            'pi_E_E': p[4]
        }

        # Handle optional rho (finite source) if vector length matches
        if len(p) > 5 and p[5] > 0.0:
            mm_params['rho'] = p[5]

        # MulensModel handles the complex parallax and Earth orbital motion
        model = mm.Model(parameters=mm_params, coords=self.coords)
        mags = model.get_magnification(self.times, satellite_skycoord=self.obs_skycoord)

        outputs[0][0] = np.array(mags, dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_grads):
        params = inputs[0]
        g = output_grads[0]
        return [MulensMagGradOp(self.times, self.coords)(params, g)]


class MulensMagGradOp(Op):
    """Numerical gradient for MulensMagOp"""
    itypes = [pt.dvector, pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, times, coords=None, eps=1e-6):
        self.times = times
        self.coords = coords
        self.eps = eps

    def perform(self, node, inputs, outputs):
        params, g = inputs
        out = np.zeros_like(params)

        # Baseline
        f_x = self._calc(params)

        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += self.eps
            f_plus = self._calc(p_plus)

            diff = (f_plus - f_x) / self.eps
            out[i] = np.sum(g * diff)

        outputs[0][0] = out

    def _calc(self, p):
        mm_params = {'t_0': p[0], 'u_0': p[1], 't_E': p[2], 'pi_E_N': p[3], 'pi_E_E': p[4]}
        if len(p) > 5 and p[5] > 0.0:
            mm_params['rho'] = p[5]
        model = mm.Model(parameters=mm_params, coords=self.coords)
        return model.get_magnification(self.times, satellite_skycoord=self.obs_skycoord)