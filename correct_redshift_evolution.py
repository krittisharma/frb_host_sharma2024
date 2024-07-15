# standard regular-use python packages
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# standard imports for my work
from bpt_utils import *
from read_transients_data import *
from generate_bkg_galaxies import *
from helper_functions import *


# Necessary functions for generating background galaxy population
nlogm, nsfr, dz = 40, 40, 0.001
mmin, mmax = 5, 12
sfrmin, sfrmax = -5, 3


def sfr_z(z, mdl='2017'):
    """
    Calculate the star formation rate as a function of redshift.

    Parameters:
    ----------
    z   : float
        Redshift.
    mdl : str
        Model for SFR calculation ('2017' or '2014').

    Returns:
    -------
    SFR : float
        Star formation rate in units of M_sun / Mpc^3 / yr.
    """
    if mdl == '2017':
        # SFR model from Madau & Fragos 2017
        return 0.01 * (1 + z)**2.6 / (1 + ((1 + z) / 3.2)**6.2)
    elif mdl == '2014':
        # SFR model from Madau & Dickinson 2014
        return 0.015 * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6)
    else:
        raise ValueError("Invalid model. Choose '2017' or '2014' for mdl.")



def corr_z_evol(z, logMstar, logSFR, sfr_prob_density_z0, sfr_prob_density_z, 
                mmin=mmin, mmax=mmax):
    """
    Correct for redshift evolution in stellar mass and star formation rate.

    Parameters:
    ----------
    z                   : float
        Redshift.
    logMstar            : float
        Logarithm of stellar mass.
    logSFR              : float
        Logarithm of star formation rate.
    sfr_prob_density_z0 : array_like
        Probability density of SFR at redshift z=0.
    sfr_prob_density_z  : array_like
        Probability density of SFR at redshift z.
    mmin                : float
        Minimum mass limit.
    mmax                : float
        Maximum mass limit.

    Returns:
    -------
    Meq_z0  : float
        Equivalent stellar mass at z=0.
    new_sfr : float
        Corrected star formation rate.
    """

    logm = np.linspace(mmin, mmax, 100)

    # Interpolate SFR probability density at redshift z
    p_int_z = np.interp(logMstar, logm, sfr_prob_density_z)

    # Interpolate equivalent stellar mass at z=0
    Meq_z0 = np.interp(p_int_z, sfr_prob_density_z0, logm)

    # Calculate offset from main sequence
    offset_from_MS = logSFR - generate_ms_center(logMstar, z)

    # Calculate new star formation rate
    new_sfr = generate_ms_center(Meq_z0, 0) + offset_from_MS

    return Meq_z0, new_sfr



def generate_z_corr_samples(transient_df, Nsamp=100, mmin=mmin, mmax=mmax):
    
    """
    Generate samples of stellar mass and star formation rate for transient data, 
    corrected for redshift evolution.

    Parameters:
    ----------
    transient_df : DataFrame
        DataFrame containing transient data with columns ['z', 'logM', 'logSFR',
        'logM_errl', 'logM_erru', 'logSFR_errl', 'logSFR_erru'].
    Nsamp        : int, optional
        Number of samples to generate for each transient (default=100).
    mmin         : float
        Minimum mass limit for generating samples.
    mmax         : float
        Maximum mass limit for generating samples.

    Returns:
    -------
    transient_logM_samples_corr   : array_like
        Corrected samples of log(stellar mass) for transients.
    transient_logSFR_samples_corr : array_like
        Corrected samples of log(star formation rate) for transients.
    """
    # Generate a grid for stellar mass and background SFR at z=0
    logm = np.linspace(mmin, mmax, 100)
    phi_z0, _, _ = generate_phi(0, mmin)
    logsfr_bkg_z0 = [generate_ms_center(logm[i], 0) for i in range(len(logm))]
    sfr_bkg_density_z0 = (10**np.array(phi_z0)) * (10**np.array(logsfr_bkg_z0))
    sfr_prob_density_z0 = np.cumsum(sfr_bkg_density_z0) / \
        max(np.cumsum(sfr_bkg_density_z0))

    transient_logM_samples_corr, transient_logSFR_samples_corr = [], []

    # Iterate over each transient in the DataFrame
    for i in tqdm(range(len(transient_df))):
        # Generate grid for stellar mass and background SFR at transient redshift
        phi_z, _, _ = generate_phi(transient_df["z"][i])
        logsfr_bkg_z = [generate_ms_center(logm[k], transient_df["z"][i]) for 
                        k in range(len(logm))]
        sfr_bkg_density_z = (10**np.array(phi_z)) * (10**np.array(logsfr_bkg_z))
        sfr_prob_density_z = np.cumsum(sfr_bkg_density_z) / max(
            np.cumsum(sfr_bkg_density_z))

        # Correct for redshift evolution using corr_z_evol function
        Meq_z0, SFReq_z0 = corr_z_evol(transient_df["z"][i], 
                                       transient_df["logM"][i], 
                                       transient_df["logSFR"][i], 
                                       sfr_prob_density_z0, 
                                       sfr_prob_density_z)

        # Generate samples of log(stellar mass) and log(SFR) for each transient
        transient_logM_samples_corr.append(
            generate_samples(Meq_z0, transient_df["logM_errl"][i], 
                             transient_df["logM_erru"][i], Nsamp))
        transient_logSFR_samples_corr.append(
            generate_samples(SFReq_z0, 
                             transient_df["logSFR_errl"][i], 
                             transient_df["logSFR_erru"][i], 
                             Nsamp))

    return np.array(transient_logM_samples_corr), \
        np.array(transient_logSFR_samples_corr)


