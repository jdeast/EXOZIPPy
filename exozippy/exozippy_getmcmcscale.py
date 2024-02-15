import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from collections import OrderedDict
import copy

'''
Inputs:
    bestpars (array-like) - size n array of the best fit parameters
    chi2func (function) - function that calculates chi^2 of your model given the parameters specified by bestpars

'''


def flatten_dict(od):
    flat_od = OrderedDict()
    flat_pars = []
    dims = []
    for key, value in od.items():
        if not is_list_like(value):
            flat_od[key] = value
        else:
            for i, val in enumerate(value):
                new_key = key + '.' + str(i)
                flat_od[new_key] = val
            flat_pars.append(key)
            dims.append(len(value))
    return flat_od


def exozippy_getmcmcscale(bestpars, chi2func, tofit=None,
                          seedscale=None, bestchi2=None,
                          angular=None, debug=False,
                          skipiter=False, logname=None):
    bestpars = flatten_dict(bestpars)
    npars = len(bestpars)
    if tofit is None:
        tofit = bestpars.keys()
    nfit = len(tofit)
    if seedscale is None:
        seedscale = OrderedDict()
        for key, value in bestpars.items():
            seedscale[key] = np.zeros_like(value) + 1e-3
    elif is_list_like(seedscale) and len(seedscale) == nfit:
        if not isinstance(seedscale, OrderedDict):
            scaledict = OrderedDict()
            for i, key in enumerate(tofit):
                scaledict[key] = np.zeros_like(bestpars[key]) + seedscale[i]
            seedscale = scaledict
    maxiter = 1e4

    if bestchi2 == None:
        bestchi2 = chi2func(bestpars)

    if bestchi2 == np.inf:
        print('Best chi^2 is out of bounds; refine bestpars')
        return

    origbestchi2 = bestchi2

    par_df = pd.DataFrame(columns=['Param', 'Minimum Step', 'Maximum Step', 'Value', 'Chi^2', 'Best Chi^2'])
    if debug:
        print('{:<20} \t {:<8} \t {:<8} \t {:<10} \t {:<10} \t {:<10}'.format('Param', 'Min Step', 'Max Step', 'Value',
                                                                              'Chi^2', 'Best Chi^2'))

    mcmcscale = OrderedDict()
    for key in tofit:
        mcmcscale[key] = np.zeros(2) + seedscale[key]
    betterfound = False

    # print(mcmcscale)

    mcmcscale_avg = OrderedDict()

    for i, key in enumerate(tofit):
        bounds = np.array(list(mcmcscale[key]))
        for j in range(2):
            testpars = copy.deepcopy(bestpars)
            test_par = np.array([bestpars[key]])
            minstep = 0
            maxstep = 0
            niter = 0
            bestdeltachi2 = np.inf
            bestscale = 0
            if j == 0:
                s = '(hi)'
            else:
                s = '(lo)'

            chi2 = bestchi2 + 1
            while np.abs(chi2 - bestchi2 - 1) < 1e-8:
                chi2changed = False
                # an infinite step size means it's not constrained

                if np.isinf(test_par[niter] + bounds[j]) or np.isinf(test_par[niter] - bounds[j]):
                    print("EXOZIPPy_GETMCMCSCALE: Parameter " + str(
                        key) + " is unconstrained. Check your starting conditions")
                    bounds[j] = np.nan
                    continue

                # add the offset to a parameter
                if j == 0:
                    test_par = np.append(test_par, test_par[niter] + bounds[j])
                else:
                    test_par = np.append(test_par, test_par[niter] - bounds[j])
                # determine the new chi^2
                testpars[key] = test_par[niter + 1]
                chi2 = chi2func(testpars)

                if chi2 - bestchi2 >= 1:
                    maxstep = bounds[j]
                    bounds[j] = (maxstep + minstep) / 2
                elif chi2 - bestchi2 >= 0:
                    minstep = bounds[j]
                    if maxstep == 0:
                        bounds[j] *= 2
                    else:
                        bounds[j] = (maxstep + minstep) / 2
                else:
                    if debug:
                        print('WARNING: better chi2 found by varying parameter ' + str(
                            key) + ' from {0:.6g} to {1:.6g} ({2:.6g})'.format(test_par[niter], test_par[niter + 1],
                                                                               chi2))

                    if (origbestchi2 - bestchi2) > 1:
                        betterfound = True

                    # chi2 is actually lower! (Didn't find the best fit)
                    # attempt to fix
                    bestpars = copy.deepcopy(testpars)

                    # could be way off, double the step for faster convergence
                    bounds[j] *= 2
                    bestchi2 = chi2
                    niter = 0
                    chi2changed = True

                deltachi2 = chi2 - bestchi2

                # in case we chance upon a better match than we bracket
                # (implies chi^2 surface is rough)
                if np.abs(deltachi2 - 1) > np.abs(bestdeltachi2 - 1):
                    bestdeltachi2 = deltachi2
                    bestscale = bounds[j]

                # can't always sample fine enough to get exactly
                # deltachi2 = 1 because chi^2 surface not perfectly smooth

                if np.abs(minstep - maxstep) < 1e-12 or niter > maxiter:
                    test_par[niter + 1] = test_par[niter] - 2 * bounds[j]
                    testpars[key] = test_par[niter + 1]
                    lowboundchi2 = chi2func(testpars)
                    test_par[niter + 1] = test_par[niter] + 2 * bounds[j]
                    testpars[key] = test_par[niter + 1]
                    hiboundchi2 = chi2func(testpars)

                    if np.any(np.isinf([chi2, lowboundchi2, hiboundchi2])):
                        if j == 0:
                            bound = 'upper'
                        else:
                            bound = 'lower'

                        print('The ' + bound + ' bound for parameter ' + str(
                            key) + ' is critical; it must be physically and independently motivated.')
                        if bestscale != 0:
                            bounds[j] = bestscale / 100
                        else:
                            bounds[j] = maxstep / 100
                        chi2 = bestchi2 + 1
                        continue
                    elif not chi2changed:
                        if np.abs(bestdeltachi2 - 1) < 0.75:
                            bounds[j] = bestscale
                        else:
                            if bestdeltachi2 == 0:
                                newscale = bestscale / 100
                            else:
                                newscale = bestscale / bestdeltachi2 / 10  # better to err on the side of too small
                            print('Cannot find the value for which deltachi^2 = 1 for parameter ' + str(
                                key) + '; assuming rough chi^2 surface. Using delta chi^2 = {0} and scaling step ({1}) to {2}'.format(
                                bestdeltachi2, bestscale, newscale))

                        # extrapolate scale to delta chi^2 = 1
                        # (may cause issues near boundaries)
                        bounds[j] = newscale

                    chi2 = bestchi2 + 1

                # if the parameter has no influence on the chi2
                if abs(chi2 - bestchi2 - 1) == 1 & niter > maxiter:
                    print('ERROR: changing parameter ' + str(key) + ' does not change the chi^2. Exclude from fit.')

                # if angle is so poorly constrained
                # no value has delta chi^2 = 1
                if angular is not None and minstep > 2 * np.pi:
                    print('WARNING: no constraint on angle')
                    bounds[j] = np.pi
                    chi2 = bestchi2 + 1

                if debug:
                    print('{:<20s} \t {:<8.4g} \t {:<8.4g} \t {:<10.6g} \t {:<10.6g} \t {:<10.6g}'.format(key + ' ' + s,
                                                                                                          minstep,
                                                                                                          maxstep,
                                                                                                          test_par[
                                                                                                              niter + 1],
                                                                                                          chi2,
                                                                                                          bestchi2))

                niter += 1
            par_df.loc[len(par_df.index)] = [key + ' ' + s, minstep, maxstep, test_par[niter], chi2, bestchi2]

        mcmcscale[key] = bounds
        mcmcscale_avg[key] = np.sum(np.array(list(mcmcscale[key]))) / 2
        # print(par_df)
    if debug:
        print(par_df)

    if betterfound and not skipiter:
        print(
            'Better Chi^2 found ({0:.6g} vs {1:.6g}). Restarting EXOFAST_GETMCMCSCALE at new best fit'.format(bestchi2,
                                                                                                              origbestchi2))
        print('This means the best fit was not properly identified and MCMC results may be slow or suspect!')
        print(
            'If the chi^2 difference is large, you may want to revise your starting parameters and/or check your priors, then restart')
        print('Proceed with caution and skepticism')
        return exozippy_getmcmcscale(bestpars, chi2func, tofit=tofit, seedscale=seedscale, bestchi2=bestchi2,
                                     angular=angular, debug=debug, skipiter=skipiter)

    # for i in range(2):
    #     bad = np.where(np.logical_or(np.isinf(mcmcscale[i]), mcmcscale[i] == 0))[0]
    #     if len(bad) != 0:
    #         mcmcscale[i, bad] = mcmcscale[1 - i, bad]
    # bad = np.where(np.isinf(mcmcscale), mcmcscale == 0)[0]

    return mcmcscale_avg, bestpars