# standard regular-use python packages
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# standard imports for my work
from bpt_utils import *
from read_transients_data import *
from correct_redshift_evolution import *
from generate_bkg_galaxies import *
from helper_functions import *

# Necessary functions for generating background galaxy population
nlogm, nsfr, dz = 40, 40, 0.001
mmin, mmax = 5, 12
sfrmin, sfrmax = -5, 3



def print_sigmas(arr, sigma):
    """
    Calculate percentiles for given sigma values relative to the median.

    Parameters:
    ----------
    arr : array-like
        Input array from which percentiles are computed.
    sigma : int
        Sigma value to determine the percentile ranges.

    Returns:
    -------
    tuple
        Depending on `sigma`, returns (median, lower_sigma, upper_sigma) tuples.
        Returns (0, 0, 0) if `sigma` is not 1, 2, or 3.
    """
    if sigma == 1:
        # Return median, 1-sigma lower bound, and 1-sigma upper bound
        return (
            round(np.percentile(arr, 50), 3),
            round(np.percentile(arr, 15.9) - np.percentile(arr, 50), 3),
            round(np.percentile(arr, 84.1) - np.percentile(arr, 50), 3)
        )
    elif sigma == 2:
        # Return median, 2-sigma lower bound, and 2-sigma upper bound
        return (
            np.percentile(arr, 50),
            np.percentile(arr, 2.3) - np.percentile(arr, 50),
            np.percentile(arr, 97.7) - np.percentile(arr, 50)
        )
    elif sigma == 3:
        # Return median, 3-sigma lower bound, and 3-sigma upper bound
        return (
            np.percentile(arr, 50),
            np.percentile(arr, 0.1) - np.percentile(arr, 50),
            np.percentile(arr, 99.9) - np.percentile(arr, 50)
        )
    else:
        # Return zeros if sigma is not 1, 2, or 3
        return 0, 0, 0
    


def return_quants(outs, sigma):
    """
    Calculate and return the 16th, 50th, and 84th percentiles (quantiles) of 
    each column in the `outs` array.

    Parameters:
    ----------
    outs : array-like
        2D array containing the data from which quantiles are computed.
    sigma : float
        Standard deviation value for calculating percentiles.

    Returns:
    -------
    quant16 : list
        List of 16th percentiles for each column.
    quant50 : list
        List of 50th percentiles (median) for each column.
    quant84 : list
        List of 84th percentiles for each column.
    """
    quant16, quant50, quant84 = [], [], []
    
    for j in range(np.shape(outs)[1]):
        arr = []
        for i in range(np.shape(outs)[0]):
            arr.append(outs[i][j])
        
        quantiles = print_sigmas(arr, sigma)
        quant16.append(quantiles[1]), quant16.append(quantiles[1])
        quant50.append(quantiles[0]), quant50.append(quantiles[0])
        quant84.append(quantiles[2]), quant84.append(quantiles[2])
    
    return quant16, quant50, quant84

    

def return_quants_single(outs, sigma=1):
    """
    Compute quantiles from outputs.

    Parameters:
    ----------
    outs : array-like
        Array of outputs.
    sigma : float
        Sigma value for quantiles.

    Returns:
    -------
    quant16 : array-like
        16th percentile for each column of `outs`.
    quant50 : array-like
        50th percentile (median) for each column of `outs`.
    quant84 : array-like
        84th percentile for each column of `outs`.
    """
    quant16, quant50, quant84 = [], [], []
    for j in range(np.shape(outs)[1]):
        arr = []
        for i in range(np.shape(outs)[0]):
            arr.append(outs[i][j])
        quant16.append(print_sigmas(arr, sigma)[1])  # Assuming print_sigmas calculates percentiles
        quant50.append(print_sigmas(arr, sigma)[0])
        quant84.append(print_sigmas(arr, sigma)[2])
    return quant16, quant50, quant84



def generate_samples(mean, high, low, Nsamp):
    """
    Generate samples from a normal distribution with varying standard deviations.

    Parameters:
    ----------
    mean  : float
        Mean of the normal distribution.
    high  : float
        Standard deviation for samples above the mean.
    low   : float
        Standard deviation for samples below the mean.
    Nsamp : int
        Number of samples to generate.

    Returns:
    -------
    samples : list
        List of generated samples.
    """
    try:
        high, low = np.abs(high), np.abs(low)  # Ensure positive values for standard deviations
    except:
        high, low = 0, 0  # Set to zero if conversion fails (unlikely case)

    samples = []
    for j in range(Nsamp):
        toss = np.random.uniform(0, 1)  # Random toss to determine sampling direction
        if toss <= 0.5:
            sample = np.random.normal(mean, low)  # Sample from lower standard deviation
            if sample > mean:
                samples.append(mean - (sample - mean))  # Reflect sample if it exceeds mean
            else:
                samples.append(sample)
        else:
            sample = np.random.normal(mean, high)  # Sample from higher standard deviation
            if sample < mean:
                samples.append(mean + (mean - sample))  # Reflect sample if it falls below mean
            else:
                samples.append(sample)

    return samples



def generate_equidistant_offsets_ellipse(ra_off_center, dec_off_center, 
                                         ra_off_center_err, dec_off_center_err, 
                                         toss):
    """
    Generate equidistant offsets from a center in sky coordinates within an 
    ellipse defined by uncertainties.

    Parameters:
    ----------
    ra_off_center : float
        Right ascension offset from the center.
    dec_off_center : float
        Declination offset from the center.
    ra_off_center_err : float
        Uncertainty in the right ascension offset.
    dec_off_center_err : float
        Uncertainty in the declination offset.
    toss : int
        Flag indicating whether to generate offsets closer (0) or 
        further away (1) from the center.

    Returns:
    -------
    offset : Generated equidistant offset from the center within the defined 
             ellipse.
    """
    dist = (ra_off_center**2 + dec_off_center**2)**0.5
    mean = [np.abs(ra_off_center), np.abs(dec_off_center)]
    covariance = [[ra_off_center_err**2, 0], [0, dec_off_center_err**2]]
    condition = True
    while condition:
        samples = np.random.multivariate_normal(mean, covariance, 1)
        offset = (samples[0][0]**2 + samples[0][1]**2)**0.5
        if (offset >= dist) and (toss == 1): # want it to be further
            condition = False
        elif (offset < dist) and (toss == 0): # want it to be closer
            condition = False
        else:
            condition = True
    return offset



def generate_equidistant_offsets_circle(offset, offset_err, toss):
    """
    Generate equidistant offsets from a center within a circle defined by an 
    offset and uncertainty.

    Parameters:
    ----------
    offset     : float
        Offset from the center.
    offset_err : float
        Uncertainty in the offset.
    toss       : int
        Flag indicating whether to generate offsets closer (0) or 
        further away (1) from the center.

    Returns:
    -------
    offset : Generated equidistant offset from the center within the defined 
             ellipse.
    """
    condition = True
    while condition:
        sample_offset = np.random.normal(offset, offset_err, 1)[0]
        if (sample_offset >= offset) and (toss == 1): # want it to be further
            condition = False
        elif (sample_offset < offset) and (toss == 0): # want it to be closer
            condition = False
        else:
            condition = True
    return [sample_offset]
    

