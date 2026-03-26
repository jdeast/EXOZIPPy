import numpy as np

# Persistent track storage
tracks = None

def massradius_mist(mstar, feh, age, teff, rstar, vvcrit=None, alpha=None, span=1, epsname=None, debug=False,
                    gravitysun=27420.011, fitage=False, ageweight=None, verbose=False, logname=None, 
                    trackfile=None, allowold=False, tefffloor=None, fehfloor=None, rstarfloor=None, 
                    agefloor=None, pngname=None, range=None):
    '''
    ;+
; NAME:
;   massradius_mist
;
; PURPOSE: 
;   Interpolate the MIST stellar evolutionary models to derive Teff
;   and Rstar from mass, metallicity, and age. Intended to be a drop in
;   replacement for the Yonsie Yale model interpolation
;   (massradius_yy3.pro).
;
; CALLING SEQUENCE:
;   chi2 = massradius_mist(mstar, feh, age, teff, rstar, $
;                          VVCRIT=vvcrit, ALPHA=alpha, SPAN=span,$
;                          MISTRSTAR=mistrstar, MISTTEFF=mistteff)
; INPUTS:
;
;    MSTAR  - The mass of the star, in m_sun
;    FEH    - The metallicity of the star [Fe/H]
;    AGE    - The age of the star, in Gyr
;    RSTAR  - The radius you expect; used to calculate a chi^2
;    TEFF   - The Teff you expect; used to calculate a chi^2
;    
; OPTIONAL INPUTS:
;   VVCRIT    - The rotational velocity normalized by the critical
;               rotation speed. Must be 0.0d0 or 0.4d0 (default 0.0d0).
;   ALPHA     - The alpha abundance. Must be 0.0 (default 0.0). A
;               placeholder for future improvements to MIST models.
;   SPAN      - The interpolation is done at the closest value +/-
;               SPAN grid points in the evolutionary tracks in mass,
;               age, metallicity. The larger this number, the longer it
;               takes. Default=1. Change with care.
;   EPSNAME   - A string specifying the name of postscript file to plot
;               the evolutionary track. If not specified, no plot is
;               generated.
;
; OPTIONAL KEYWORDS:
;   DEBUG     - If set, will plot the teff and rstar over the MIST
;               Isochrone.
;
; OPTIONAL OUTPUTS:
;   MISTRSTAR - The rstar interpolated from the MIST models.
;   MISTTEFF  - The Teff interpolated from the MIST models.
;
; RESULT:
;   The chi^2 penalty due to the departure from the MIST models,
;   assuming 3% errors in the MIST model values.
;
; COMMON BLOCKS:
;   MIST_BLOCK:
;     Loading EEPs (model tracks) is very slow. This common block
;     allows us to store the tracks in memory between calls. The first
;     call will take ~3 seconds. Subsequent calls that use the same
;     EEP files take 1 ms.
;
; EXAMPLE: 
;   ;; penalize a model for straying from the MIST models 
;   chi2 += massradius_mist(mstar, feh, age, rstar=rstar, teff=teff)
;
; MODIFICATION HISTORY
; 
;  2018/01 -- Written, JDE
;-
    '''
    global tracks

    if tefffloor is None:
        tefffloor = -1
    if fehfloor is None:
        fehfloor = -1
    if rstarfloor is None:
        rstarfloor = -1
    if agefloor is None:
        agefloor = -1

    if tracks is None:
        # Initialize the grids
        allowedmass = np.array([
            0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20,
            1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52,
            1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84,
            1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98, 2.00, 2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14, 2.16,
            2.18, 2.20, 2.22, 2.24, 2.26, 2.28, 2.30, 2.32, 2.34, 2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48,
            2.50, 2.52, 2.54, 2.56, 2.58, 2.60, 2.62, 2.64, 2.66, 2.68, 2.70, 2.72, 2.74, 2.76, 2.78, 2.80,
            3.00, 3.20, 3.40, 3.60, 3.80, 4.00, 4.20, 4.40, 4.60, 4.80, 5.00, 5.20, 5.40, 5.60, 5.80, 6.00,
            6.20, 6.40, 6.60, 6.80, 7.00, 7.20, 7.40, 7.60, 7.80, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00,
            14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00, 22.00, 24.00, 26.00, 28.00, 30.00, 32.00,
            34.00, 36.00, 38.00, 40.00, 45.00, 50.00, 55.00, 60.00, 65.00, 70.00, 75.00, 80.00, 85.00,
            90.00, 95.00, 100.00, 105.00, 110.00, 115.00, 120.00, 125.00, 130.00, 135.00, 140.00, 145.00,
            150.00, 175.00, 200.00, 225.00, 250.00, 275.00, 300.00
        ])

        allowedinitfeh = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.25,
                                   -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5])
        allowedvvcrit = np.array([0.0, 0.4])
        allowedalpha = np.array([0.0])

        nmass = len(allowedmass)
        nfeh = len(allowedinitfeh)
        nvvcrit = len(allowedvvcrit)
        nalpha = len(allowedalpha)

        tracks = np.empty((nmass, nfeh, nvvcrit, nalpha), dtype=object)

    if not (0.1 <= mstar <= 300):
        if verbose:
            print(f"Mstar ({mstar}) is out of range [0.1, 300]", file=logname or None)
        return np.inf

    if not (-4.0 <= feh <= 0.5):
        if verbose:
            print(f"initfeh ({feh}) is out of range [-4, 0.5]", file=logname or None)
        return np.inf

    massndx = np.argmin(np.abs(allowedmass - mstar))
    fehndx = np.argmin(np.abs(allowedinitfeh - feh))

    vvcritndx = 0 if vvcrit is None else np.where(allowedvvcrit == vvcrit)[0][0]
    alphandx = 0 if alpha is None else np.where(allowedalpha == alpha)[0][0]

    if tracks[massndx, fehndx, vvcritndx, alphandx] is None:
        tracks[massndx, fehndx, vvcritndx, alphandx] = readeep(
            allowedmass[massndx],
            allowedinitfeh[fehndx],
            vvcrit=allowedvvcrit[vvcritndx],
            alpha=allowedalpha[alphandx]
        )

    data = tracks[massndx, fehndx, vvcritndx, alphandx]
    ages, rstars, teffs, fehs, ageweights_data = data

    eep = np.searchsorted(ages, age)
    if eep == len(ages):
        eep -= 1
    if eep < 1:
        if verbose:
            print(f"EEP ({eep}) is out of range [1, ∞]", file=logname or None)
        return np.inf

    neep = len(ages)
    if eep >= neep:
        if verbose:
            print(f"EEP ({eep}) is out of bounds for track with {neep} points", file=logname or None)
        return np.inf

    # Interpolation using two closest ages
    x_eep = (age - ages[eep - 1]) / (ages[eep] - ages[eep - 1]) if ages[eep] != ages[eep - 1] else 0.0
    mistage = (1 - x_eep) * ages[eep - 1] + x_eep * ages[eep]
    mistrstar = (1 - x_eep) * rstars[eep - 1] + x_eep * rstars[eep]
    mistteff = (1 - x_eep) * teffs[eep - 1] + x_eep * teffs[eep]
    mistfeh = (1 - x_eep) * fehs[eep - 1] + x_eep * fehs[eep]
    ageweight_interp = (1 - x_eep) * ageweights_data[eep - 1] + x_eep * ageweights_data[eep]

    if mistage < 0 or (not allowold and mistage > 13.82):
        if verbose:
            print(f"Age ({mistage}) is out of range", file=logname or None)
        return np.inf

    percenterror = 0.03 - 0.025 * np.log10(mstar) + 0.045 * (np.log10(mstar))**2

    chi2_rstar = ((mistrstar - rstar) / (rstarfloor * mistrstar if rstarfloor > 0 else percenterror * mistrstar))**2
    chi2_teff = ((mistteff - teff) / (tefffloor * mistteff if tefffloor > 0 else percenterror * mistteff))**2
    chi2_feh = ((mistfeh - feh) / (fehfloor if fehfloor > 0 else percenterror))**2
    chi2_age = ((mistage - age) / (agefloor * mistage if agefloor > 0 else percenterror * mistage))**2

    chi2 = chi2_rstar + chi2_teff + chi2_feh + chi2_age
    return chi2
