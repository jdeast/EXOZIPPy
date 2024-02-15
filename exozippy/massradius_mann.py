import numpy as np

def massradius_mann(ks0, feh=None, distance=None):
    # Compute the absolute magnitude if given the apparent magnitude
    if distance is None:
        ks = ks0
    else:
        ks = ks0 - 2.5 * np.log10((distance / 10.0) ** 2)
    
    # Mann+ 2015 to get Rstar
    if feh is None:
        # eq 4, table 1
        ai = [1.9515, -0.3520, 0.01680]
        rstar = ai[0] + ai[1] * ks + ai[2] * ks ** 2
        sigma_rstar = rstar * 0.0289
    else:
        # eq 5, table 1
        ai = [1.9305, -0.3466, 0.01647, 0.04458]
        rstar = ai[0] + ai[1] * ks + ai[2] * ks ** 2 + ai[3] * feh
        sigma_rstar = rstar * 0.027
    
    # Mann+ 2019 to get mass
    zp = 7.5
    if feh is None:
        # eq 4, table 6 (n=5)
        bi = [-0.642, -0.208, -8.43e-4, 7.87e-3, 1.42e-4, -2.13e-4]
        mstar = 10 ** (bi[0] + bi[1] * (ks - zp) + bi[2] * (ks - zp) ** 2 +
                       bi[3] * (ks - zp) ** 3 + bi[4] * (ks - zp) ** 4 + 
                       bi[4] * (ks - zp) ** 5)
        sigma_mstar = mstar * 0.020
    else:
        # eq 5, table 6 (n=5)
        bi = [-0.647, -0.207, -6.53e-4, 7.13e-3, 1.84e-4, -1.6e-4, -0.0035]
        mstar = (1.0+feh*bi[6])*10**(bi[0] + bi[1]*(ks - zp) + bi[2]*(ks - zp)**2 +
                                     bi[3]*(ks - zp)**3 + bi[4]*(ks-zp)**4 +
                                     bi[4]*(ks - zp)**5)
        sigma_mstar = mstar * 0.021
    
    return mstar, rstar, sigma_mstar, sigma_rstar
