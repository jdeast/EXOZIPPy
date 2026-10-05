# Vendored VERBATIM below this header from
#   https://github.com/avanderburg/keplersplinev2
#   keplersplinev2.py at commit e105456e964b831441f90ba6aafa855a46dd7ea9
#   (2024-04-17)
# Andrew Vanderburg's Python Kepler spline. Used ONLY by the test suite, as an
# independent reference for components/transit/spline.py (a port of the IDL
# keplerspline that EXOFASTv2 bundles): tests/test_transit_fitspline.py.
# Do not edit below the header -- re-vendor instead. Excluded from ruff
# (pyproject.toml [tool.ruff] extend-exclude). Needs pydl (dev dependency).
#
# MIT License
#
# Copyright (c) 2020 avanderburg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Functions for computing normalization splines for Kepler light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
from pydl.pydlutils import bspline

#from third_party.robust_mean import robust_mean


class InsufficientPointsError(Exception):
  """Indicates that insufficient points were available for spline fitting."""
  pass


class SplineError(Exception):
  """Indicates an error in the underlying spline-fitting implementation."""
  pass

def split(all_time, all_flux, gap_width=0.75):
  """Splits a light curve on discontinuities (gaps).
  This function accepts a light curve that is either a single segment, or is
  piecewise defined (e.g. split by quarter breaks or gaps in the in the data).
  Args:
    all_time: Numpy array or sequence of numpy arrays; each is a sequence of
      time values.
    all_flux: Numpy array or sequence of numpy arrays; each is a sequence of
      flux values of the corresponding time array.
    gap_width: Minimum gap size (in time units) for a split.
  Returns:
    out_time: List of numpy arrays; the split time arrays.
    out_flux: List of numpy arrays; the split flux arrays.
  """
  # Handle single-segment inputs.
  if isinstance(all_time, np.ndarray) and all_time.ndim == 1:
    all_time = [all_time]
    all_flux = [all_flux]

  out_time = []
  out_flux = []
  for time, flux in zip(all_time, all_flux):
    start = 0
    for end in range(1, len(time) + 1):
      # Choose the largest endpoint such that time[start:end] has no gaps.
      if end == len(time) or time[end] - time[end - 1] > gap_width:
        out_time.append(time[start:end])
        out_flux.append(flux[start:end])
        start = end

  return out_time, out_flux


def robust_mean(y, cut):
  """Computes a robust mean estimate in the presence of outliers.
  Args:
    y: 1D numpy array. Assumed to be normally distributed with outliers.
    cut: Points more than this number of standard deviations from the median are
      ignored.
  Returns:
    mean: A robust estimate of the mean of y.
    mean_stddev: The standard deviation of the mean.
    mask: Boolean array with the same length as y. Values corresponding to
        outliers in y are False. All other values are True.
  """
  # First, make a robust estimate of the standard deviation of y, assuming y is
  # normally distributed. The conversion factor of 1.4826 takes the median
  # absolute deviation to the standard deviation of a normal distribution.
  # See, e.g. https://www.mathworks.com/help/stats/mad.html.
  absdev = np.abs(y - np.median(y))
  sigma = 1.4826 * np.median(absdev)

  # If the previous estimate of the standard deviation using the median absolute
  # deviation is zero, fall back to a robust estimate using the mean absolute
  # deviation. This estimator has a different conversion factor of 1.253.
  # See, e.g. https://www.mathworks.com/help/stats/mad.html.
  if sigma < 1.0e-24:
    sigma = 1.253 * np.mean(absdev)

  # Identify outliers using our estimate of the standard deviation of y.
  mask = absdev <= cut * sigma

  # Now, recompute the standard deviation, using the sample standard deviation
  # of non-outlier points.
  sigma = np.std(y[mask])

  # Compensate the estimate of sigma due to trimming away outliers. The
  # following formula is an approximation, see
  # http://w.astro.berkeley.edu/~johnjohn/idlprocs/robust_mean.pro.
  sc = np.max([cut, 1.0])
  if sc <= 4.5:
    sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

  # Identify outliers using our second estimate of the standard deviation of y.
  mask = absdev <= cut * sigma

  # Now, recompute the standard deviation, using the sample standard deviation
  # with non-outlier points.
  sigma = np.std(y[mask])

  # Compensate the estimate of sigma due to trimming away outliers.
  sc = np.max([cut, 1.0])
  if sc <= 4.5:
    sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * sc**2 + 0.020142 * sc**3)

  # Final estimate is the sample mean with outliers removed.
  mean = np.mean(y[mask])
  mean_stddev = sigma / np.sqrt(len(y) - 1.0)

  return mean, mean_stddev, mask

def kepler_spline(time, flux, bkspace=1.5, maxiter=5, outlier_cut=3, input_mask = None):
  """Computes a best-fit spline curve for a light curve segment.

  The spline is fit using an iterative process to remove outliers that may cause
  the spline to be "pulled" by discrepant points. In each iteration the spline
  is fit, and if there are any points where the absolute deviation from the
  median residual is at least 3*sigma (where sigma is a robust estimate of the
  standard deviation of the residuals), those points are removed and the spline
  is re-fit.

  Args:
    time: Numpy array; the time values of the light curve.
    flux: Numpy array; the flux (brightness) values of the light curve.
    bkspace: Spline break point spacing in time units.
    maxiter: Maximum number of attempts to fit the spline after removing badly
      fit points.
    outlier_cut: The maximum number of standard deviations from the median
      spline residual before a point is considered an outlier.

  Returns:
    spline: The values of the fitted spline corresponding to the input time
        values.
    mask: Boolean mask indicating the points used to fit the final spline.

  Raises:
    InsufficientPointsError: If there were insufficient points (after removing
        outliers) for spline fitting.
    SplineError: If the spline could not be fit, for example if the breakpoint
        spacing is too small.
  """
  if len(time) < 4:
    raise InsufficientPointsError(
        "Cannot fit a spline on less than 4 points. Got {} points.".format(
            len(time)))

  # Rescale time into [0, 1].
  t_min = np.min(time)
  t_max = np.max(time)
  time = (time - t_min) / (t_max - t_min)
  bkspace /= (t_max - t_min)  # Rescale bucket spacing.

  # Values of the best fitting spline evaluated at the time points.
  spline = None

  # Mask indicating the points used to fit the spline.
  mask = None

  if np.all(input_mask == None): input_mask = np.ones_like(time, dtype=bool)

  for _ in range(maxiter):
    if spline is None:
      mask = input_mask  # Try to fit all points, or at least the ones in our input mask.
    else:
      # Choose points where the absolute deviation from the median residual is
      # less than outlier_cut*sigma, where sigma is a robust estimate of the
      # standard deviation of the residuals from the previous spline.
      residuals = flux - spline
      new_mask = robust_mean(residuals, cut=outlier_cut)[2]
      new_mask = np.logical_and(new_mask, input_mask)
      if np.all(new_mask == mask):
        break  # Spline converged.

      mask = new_mask

    if np.sum(mask) < 4:
      # Fewer than 4 points after removing outliers. We could plausibly return
      # the spline from the previous iteration because it was fit with at least
      # 4 points. However, since the outliers were such a significant fraction
      # of the curve, the spline from the previous iteration is probably junk,
      # and we consider this a fatal error.
      raise InsufficientPointsError(
          "Cannot fit a spline on less than 4 points. After removing "
          "outliers, got {} points.".format(np.sum(mask)))

    try:
      with warnings.catch_warnings():
        # Suppress warning messages printed by pydlutils.bspline. Instead we
        # catch any exception and raise a more informative error.
        warnings.simplefilter("ignore")

        # Fit the spline on non-outlier points.
        curve = bspline.iterfit(time[mask], flux[mask], bkspace=bkspace)[0]

      # Evaluate spline at the time points.
      spline = curve.value(time)[0]
    except (IndexError, TypeError, ValueError) as e:
      raise SplineError(
          "Fitting spline failed with error: '{}'. This might be caused by the "
          "breakpoint spacing being too small, and/or there being insufficient "
          "points to fit the spline in one of the intervals.".format(e))

  return spline, mask


class SplineMetadata(object):
  """Metadata about a spline fit.

  Attributes:
    light_curve_mask: List of boolean numpy arrays indicating which points in
      the light curve were used to fit the best-fit spline.
    bkspace: The break-point spacing used for the best-fit spline.
    bad_bkspaces: List of break-point spacing values that failed.
    likelihood_term: The likelihood term of the Bayesian Information Criterion;
      -2*ln(L), where L is the likelihood of the data given the model.
    penalty_term: The penalty term for the number of parameters in the Bayesian
      Information Criterion.
    bic: The value of the Bayesian Information Criterion; equal to
      likelihood_term + penalty_coeff * penalty_term.
  """

  def __init__(self):
    self.light_curve_mask = None
    self.input_light_curve_mask = None
    self.bkspace = None
    self.bad_bkspaces = []
    self.likelihood_term = None
    self.penalty_term = None
    self.bic = None


def choose_kepler_spline(all_time,
                         all_flux,
                         bkspaces,
                         maxiter=5,
                         penalty_coeff=1.0,
                         verbose=True,
                         all_input_mask=None):
  """Computes the best-fit Kepler spline across a break-point spacings.

  Some Kepler light curves have low-frequency variability, while others have
  very high-frequency variability (e.g. due to rapid rotation). Therefore, it is
  suboptimal to use the same break-point spacing for every star. This function
  computes the best-fit spline by fitting splines with different break-point
  spacings, calculating the Bayesian Information Criterion (BIC) for each
  spline, and choosing the break-point spacing that minimizes the BIC.

  This function assumes a piecewise light curve, that is, a light curve that is
  divided into different segments (e.g. split by quarter breaks or gaps in the
  in the data). A separate spline is fit for each segment.

  Args:
    all_time: List of 1D numpy arrays; the time values of the light curve.
    all_flux: List of 1D numpy arrays; the flux values of the light curve.
    bkspaces: List of break-point spacings to try.
    maxiter: Maximum number of attempts to fit each spline after removing badly
      fit points.
    penalty_coeff: Coefficient of the penalty term for using more parameters in
      the Bayesian Information Criterion. Decreasing this value will allow more
      parameters to be used (i.e. smaller break-point spacing), and vice-versa.
    verbose: Whether to log individual spline errors. Note that if bkspaces
      contains many values (particularly small ones) then this may cause logging
      pollution if calling this function for many light curves.

  Returns:
    spline: List of numpy arrays; values of the best-fit spline corresponding to
        to the input flux arrays.
    metadata: Object containing metadata about the spline fit.
  """
  # Initialize outputs.
  best_spline = None
  metadata = SplineMetadata()

  # Compute the assumed standard deviation of Gaussian white noise about the
  # spline model. We assume that each flux value f[i] is a Gaussian random
  # variable f[i] ~ N(s[i], sigma^2), where s is the value of the true spline
  # model and sigma is the constant standard deviation for all flux values.
  # Moreover, we assume that s[i] ~= s[i+1]. Therefore,
  # (f[i+1] - f[i]) / sqrt(2) ~ N(0, sigma^2).
  scaled_diffs = [np.diff(f) / np.sqrt(2) for f in all_flux]
  scaled_diffs = np.concatenate(scaled_diffs) if scaled_diffs else np.array([])
  if not scaled_diffs.size:
    best_spline = [np.array([np.nan] * len(f)) for f in all_flux]
    metadata.light_curve_mask = [
        np.zeros_like(f, dtype=bool) for f in all_flux
    ]
    return best_spline, metadata

  # Compute the median absolute deviation as a robust estimate of sigma. The
  # conversion factor of 1.48 takes the median absolute deviation to the
  # standard deviation of a normal distribution. See, e.g.
  # https://www.mathworks.com/help/stats/mad.html.
  sigma = np.median(np.abs(scaled_diffs)) * 1.48
    
    
  #Now if we don't input any input mask we need to create a set of input masks that are all true
  if np.all(all_input_mask == None): 
        all_input_mask = []
        for eachtime in all_time:
            all_input_mask.append(np.ones_like(eachtime, dtype=bool))
            
    
  for bkspace in bkspaces:
    nparams = 0  # Total number of free parameters in the piecewise spline.
    npoints = 0  # Total number of data points used to fit the piecewise spline.
    ssr = 0  # Sum of squared residuals between the model and the spline.

    spline = []
    light_curve_mask = []
    bad_bkspace = False  # Indicates that the current bkspace should be skipped.
    for time, flux, this_input_mask in zip(all_time, all_flux, all_input_mask):
      # Fit B-spline to this light-curve segment.
      try:
        spline_piece, mask = kepler_spline(
            time, flux, bkspace=bkspace, maxiter=maxiter, input_mask = this_input_mask)
      except InsufficientPointsError as e:
        # It's expected to occasionally see intervals with insufficient points,
        # especially if periodic signals have been removed from the light curve.
        # Skip this interval, but continue fitting the spline.
        if verbose:
          warnings.warn(str(e))
        spline.append(np.array([np.nan] * len(flux)))
        light_curve_mask.append(np.zeros_like(flux, dtype=bool))
        continue
      except SplineError as e:
        # It's expected to get a SplineError occasionally for small values of
        # bkspace. Skip this bkspace.
        if verbose:
          warnings.warn("Bad bkspace {}: {}".format(bkspace, e))
        metadata.bad_bkspaces.append(bkspace)
        bad_bkspace = True
        break

      spline.append(spline_piece)
      light_curve_mask.append(mask)

      # Accumulate the number of free parameters.
      total_time = np.max(time) - np.min(time)
      nknots = int(total_time / bkspace) + 1  # From the bspline implementation.
      nparams += nknots + 3 - 1  # number of knots + degree of spline - 1

      # Accumulate the number of points and the squared residuals.
      npoints += np.sum(mask)
      ssr += np.sum((flux[mask] - spline_piece[mask])**2)

    if bad_bkspace or not npoints:
      continue

    # The following term is -2*ln(L), where L is the likelihood of the data
    # given the model, under the assumption that the model errors are iid
    # Gaussian with mean 0 and standard deviation sigma.
    likelihood_term = npoints * np.log(2 * np.pi * sigma**2) + ssr / sigma**2

    # Penalty term for the number of parameters used to fit the model.
    penalty_term = nparams * np.log(npoints)

    # Bayesian information criterion.
    bic = likelihood_term + penalty_coeff * penalty_term

    if best_spline is None or bic < metadata.bic:
      best_spline = spline
      metadata.light_curve_mask = light_curve_mask
      metadata.input_light_curve_mask = all_input_mask
      metadata.bkspace = bkspace
      metadata.likelihood_term = likelihood_term
      metadata.penalty_term = penalty_term
      metadata.bic = bic

  if best_spline is None:
    # All bkspaces resulted in a SplineError, or all light curve intervals had
    # insufficient points.
    best_spline = [np.array([np.nan] * len(f)) for f in all_flux]
    metadata.light_curve_mask = [
        np.zeros_like(f, dtype=bool) for f in all_flux
    ]
    metadata.input_light_curve_mask = [
        np.zeros_like(f, dtype=bool) for f in all_flux
    ]
    

  return best_spline, metadata


def fit_kepler_spline(all_time,
                      all_flux,
                      bkspace_min=0.5,
                      bkspace_max=20,
                      bkspace_num=20,
                      maxiter=5,
                      penalty_coeff=1.0,
                      verbose=True):
  """Fits a Kepler spline with logarithmically-sampled breakpoint spacings.

  Args:
    all_time: List of 1D numpy arrays; the time values of the light curve.
    all_flux: List of 1D numpy arrays; the flux values of the light curve.
    bkspace_min: Minimum breakpoint spacing to try.
    bkspace_max: Maximum breakpoint spacing to try.
    bkspace_num: Number of breakpoint spacings to try.
    maxiter: Maximum number of attempts to fit each spline after removing badly
      fit points.
    penalty_coeff: Coefficient of the penalty term for using more parameters in
      the Bayesian Information Criterion. Decreasing this value will allow more
      parameters to be used (i.e. smaller break-point spacing), and vice-versa.
    verbose: Whether to log individual spline errors. Note that if bkspaces
      contains many values (particularly small ones) then this may cause logging
      pollution if calling this function for many light curves.

  Returns:
    spline: List of numpy arrays; values of the best-fit spline corresponding to
        to the input flux arrays.
    metadata: Object containing metadata about the spline fit.
  """
  # Logarithmically sample bkspace_num candidate break point spacings between
  # bkspace_min and bkspace_max.
  bkspaces = np.logspace(
      np.log10(bkspace_min), np.log10(bkspace_max), num=bkspace_num)

  return choose_kepler_spline(
      all_time,
      all_flux,
      bkspaces,
      maxiter=maxiter,
      penalty_coeff=penalty_coeff,
      verbose=verbose)


def choosekeplersplinev2(time,flux, bkspace_min=0.5, bkspace_max=20, bkspace_num=20, 
                         maxiter=5, verbose=True, input_mask=None, gap_width_in = None, return_metadata=False):
    if gap_width_in == None: gap_width_in = bkspace_min #If not specified use the same as the input bkspace
    if np.all(input_mask == None): input_mask = np.ones(len(time), dtype=bool)    
    all_time, all_flux = split(time, flux, gap_width=gap_width_in) 
    all_time2, all_input_mask = split(time, input_mask, gap_width=gap_width_in) 
    
    bkspaces = np.logspace(
      np.log10(bkspace_min), np.log10(bkspace_max), num=bkspace_num)
    
    spline, metadata = choose_kepler_spline(all_time, all_flux, verbose=False, 
                                            bkspaces =bkspaces, all_input_mask=all_input_mask)
    #print(len(spline))
    spline = np.concatenate(spline)
    
    metadata.light_curve_mask = np.concatenate(metadata.light_curve_mask)
    metadata.input_light_curve_mask = np.concatenate(metadata.input_light_curve_mask) # replace this line with input mask
    
    if return_metadata: return spline, metadata
    else: return spline


def keplersplinev2(time,flux, bkspace=1.5, maxiter=5, verbose=True, input_mask=None, gap_width_in = None, return_metadata=False):
    if gap_width_in == None: gap_width_in = bkspace #If not specified use the same as the input bkspace
    if np.all(input_mask == None): input_mask = np.ones(len(time), dtype=bool)    
    all_time, all_flux = split(time, flux, gap_width=gap_width_in) 
    all_time2, all_input_mask = split(time, input_mask, gap_width=gap_width_in) 
    
    spline, metadata = choose_kepler_spline(all_time, all_flux, verbose=False, 
                                            bkspaces =[bkspace], all_input_mask=all_input_mask)
    #print(len(spline))
    spline = np.concatenate(spline)
    
    metadata.light_curve_mask = np.concatenate(metadata.light_curve_mask)
    metadata.input_light_curve_mask = np.concatenate(metadata.input_light_curve_mask) # replace this line with input mask
    
    if return_metadata: return spline, metadata
    else: return spline



    
    
    
    
    
