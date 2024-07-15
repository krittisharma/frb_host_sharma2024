# standard regular-use python packages
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# packages needed to use Leja et al. (2022) normalizing flows results
import torch
import torch_light
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.stats import norm as norm_density

# standard imports for my work
from bpt_utils import *
from read_transients_data import *
from correct_redshift_evolution import *
from helper_functions import *

# Necessary functions for generating background galaxy population
nlogm, nsfr, dz = 40, 40, 0.001
mmin, mmax = 5, 12
sfrmin, sfrmax = -5, 3


def schechter(logm, logphi, logmstar, alpha, m_lower=None):
    """
    Generate a Schechter function.

    Parameters:
    ----------
    logm     : array_like
        Logarithm of mass values to evaluate the Schechter function.
    logphi   : float
        Logarithm of the amplitude parameter phi in the Schechter function.
    logmstar : float
        Logarithm of the characteristic mass parameter m_star.
    alpha    : float
        Slope parameter in the Schechter function.
    m_lower  : float, optional
        Lower limit of mass values.

    Returns:
    -------
    phi      : array_like
        Array of Schechter function values evaluated at logm.
    """
    # Calculate phi using the Schechter function formula
    phi = (10**logphi) * np.log(10) * (10**(logm - logmstar))**(alpha + 1) \
      * np.exp(-10**(logm - logmstar))
    
    # Apply lower mass limit if specified
    if m_lower is not None:
        phi[logm < m_lower] = 0.0
    
    return phi


    
def parameter_at_z0(y, z0, z1=0.2, z2=1.6, z3=3.0):
    """
    Compute parameter at redshift 'z0' as a function
    of the polynomial parameters 'y' and the
    redshift anchor points 'z1', 'z2', and 'z3'.

    Parameters:
    ----------
    y  : tuple
        Tuple of polynomial parameters (y1, y2, y3).
    z0 : float
        Redshift at which to compute the parameter.
    z1 : float, optional
        Redshift anchor point 1 (default is 0.2).
    z2 : float, optional
        Redshift anchor point 2 (default is 1.6).
    z3 : float, optional
        Redshift anchor point 3 (default is 3.0).

    Returns:
    -------
    float
        Parameter value at redshift z0 computed using polynomial interpolation.
    """
    y1, y2, y3 = y
    a = (( (y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3) ) /
         (z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)))
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    return a * z0**2 + b * z0 + c



def generate_phi(z0, mlim=mmin, ndraw=1000):
    """
    Generate stellar mass function for a given redshift 'z0'.

    Parameters:
    ----------
    z0    : float
        Redshift at which to generate Schechter functions.
    mlim  : float, optional
        Minimum mass limit (default is mmin).
    ndraw : int, optional
        Number of draws for parameter sampling (default is 1000).

    Returns:
    -------
    phi_50 : array_like
        Median Schechter function values.
    phi_84 : array_like
        Upper 84th percentile Schechter function values.
    phi_16 : array_like
        Lower 16th percentile Schechter function values.
    """

    # Continuity model median parameters + 1-sigma uncertainties.
    pars = {'logphi1': [-2.44, -3.08, -4.14],
            'logphi1_err': [0.02, 0.03, 0.1],
            'logphi2': [-2.89, -3.29, -3.51],
            'logphi2_err': [0.04, 0.03, 0.03],
            'logmstar': [10.79, 10.88, 10.84],
            'logmstar_err': [0.02, 0.02, 0.04],
            'alpha1': [-0.28],
            'alpha1_err': [0.07],
            'alpha2': [-1.48],
            'alpha2_err': [0.1]}
    
    draws = {}
    for par in ['logphi1', 'logphi2', 'logmstar', 'alpha1', 'alpha2']:
        samp = np.array([np.random.normal(m, e, ndraw) 
                         for m, e in zip(pars[par], pars[par+'_err'])])
        if par in ['logphi1', 'logphi2', 'logmstar']:
            draws[par] = parameter_at_z0(samp, z0)
        else:
            draws[par] = samp.squeeze()
            
    # Generate Schechter functions
    logm = np.linspace(mlim, mmax, 100)[:, None]  # log(M) grid
    phi1 = schechter(logm, draws['logphi1'], 
                     draws['logmstar'], draws['alpha1'])
    phi2 = schechter(logm, draws['logphi2'], 
                     draws['logmstar'], draws['alpha2'])
    phi = phi1 + phi2  # combined mass function
    phi_50, phi_84, phi_16 = np.percentile(phi, [50, 84, 16], axis=1)
    
    return phi_50, phi_84, phi_16



def sample_density(nlogm, nsfr, nz, ndummy, dz,
                   mgrid, sfrgrid, zgrid, dummy,
                   flow, redshift_smoothing=True):
    """
    Compute probability density from the trained flow.
    Optionally, smooth over redshift with a Gaussian kernel of dz=0.1
    to smooth over spatial inhomogeneities (recommended).

    Parameters:
    ----------
    nlogm              : int
        Number of logM bins.
    nsfr               : int
        Number of logSFR bins.
    nz                 : int
        Number of redshift bins.
    ndummy             : int
        Number of dummy variable bins.
    dz                 : float
        Redshift bin width.
    mgrid              : array_like
        Grid for logM.
    sfrgrid            : array_like
        Grid for logSFR.
    zgrid              : array_like
        Grid for redshift.
    dummy              : array_like
        Grid for dummy variable.
    flow               : object
        Trained flow model.
    redshift_smoothing : bool, optional
        Flag for redshift smoothing (default is True).

    Returns:
    -------
    prob_full          : array_like
        Probability density over the entire grid.
    """
    # Initialize probability density array
    prob_full = np.zeros(shape=(nlogm, nsfr, nz))
    
    # Loop over redshift grid and marginalize in each loop to save memory
    for i, zred in enumerate(zgrid):
        
        # Transform grid into torch-appropriate inputs
        # Order must match the order during training
        x, y, z, d = zred, mgrid - 10., sfrgrid, dummy
        x, y, z, d = np.meshgrid(x, y, z, d)
        pgrid = np.stack([x, y, z, d], axis=-1)
        pgrid.shape = (-1, 4)
        pgrid = torch.from_numpy(pgrid.astype('f4')).to('cpu')
        
        # Calculate probability (density)
        _, prior_logprob, log_det = flow(pgrid)
        prob = np.exp((prior_logprob + log_det).detach().numpy())
        
        # Reshape output to match dimensions (logM, logSFR, z, dummy)
        prob = prob.reshape(nlogm, 1, nsfr, ndummy)
        prob = np.swapaxes(prob, 1, 2)
        
        # Marginalize over 'dummy' variable
        weights = norm_density.pdf(dummy, loc=0.0, scale=1.0)
        prob_full[:, :, i] = (weights[None, None, None, :] * prob).sum(
            axis=-1).squeeze() / weights.sum()
    
    # Smooth in redshift to smooth over spatial inhomogeneities
    # Using a Gaussian kernel of width dz = 0.1
    if redshift_smoothing:
        kernel = Gaussian1DKernel(stddev=2 / dz)
        prob_smooth = np.zeros(shape=(nlogm, nsfr, nz))
        for i in range(nlogm):
            for j in range(nsfr):
                prob_smooth[i, j, :] = convolve(prob_full[i, j, :], 
                                                kernel, 
                                                boundary='extend', 
                                                preserve_nan=True)
        return prob_smooth
    else:
        return prob_full



def load_nf():
    """
    Instantiate the NormalizingFlow class and load the trained flow.

    Returns:
    -------
    flow : object
        Trained NormalizingFlow instance.
    """
    # Define relative path to trained flow
    loc = 'galaxies_data/trained_flow_nsamp100.pth'

    # Fixed dimensions of the problem
    n_units = 5
    n_dim = 4

    # Instantiate NormalizingFlow
    flow = torch_light.NormalizingFlow(n_dim, n_units)

    # Load trained flow
    state_dict = torch.load(loc)
    flow.load_state_dict(state_dict)

    return flow



def cosmos15_mass_completeness(zred):
    """
    Returns log10(stellar mass) at which the COSMOS-2015 survey is considered 
    complete at a given input redshift (zred).
    Data from Table 6 of Laigle+16, corrected to Prospector stellar mass.
    
    Parameters:
    ----------
    zred        : float
        Redshift at which to interpolate mass completeness.

    Returns:
    -------
    mcomp_prosp : float
        Interpolated log10(stellar mass) completeness.
    """
    # Redshift and corresponding mass completeness from COSMOS-2015
    zcosmos = np.array([0.175, 0.5, 0.8, 1.125, 1.525, 2.0])
    mcomp_prosp = np.array([8.57779419, 9.13212072, 9.54630419, 9.75007079,
                            10.10434753, 10.30023359])
    
    # Interpolate to find mass completeness at input redshift
    return np.interp(zred, zcosmos, mcomp_prosp)



def threedhst_mass_completeness(zred):
    """
    Returns log10(stellar mass) at which the 3D-HST survey is considered 
    complete at a given input redshift (zred).
    Data from Table 1 of Tal+14, corrected to Prospector stellar mass.
    
    Parameters:
    ----------
    zred        : float
        Redshift at which to interpolate mass completeness.

    Returns:
    -------
    mcomp_prosp : float
        Interpolated log10(stellar mass) completeness.
    """
    # Redshift and corresponding mass completeness from 3D-HST
    ztal = np.array([0.65, 1.0, 1.5, 2.1, 3.0])
    mcomp_prosp = np.array([8.86614882, 9.07108637, 9.63281923,
                            9.89486727, 10.15444536])
    
    # Interpolate to find mass completeness at input redshift
    return np.interp(zred, ztal, mcomp_prosp)



def generate_ms_center(logM_sample, z0):
    """
    Returns the center of the galaxy's star-forming main sequence at 
    redshift z0 and stellar mass logM_sample.

    Parameters:
    ----------
    logM_sample   : float
        Logarithm of stellar mass sample.
    z0            : float
        Redshift.

    Returns:
    -------
    logSFR_sample : float
        Logarithm of star formation rate sample.
    """
    # Median/Ridge parameters for the main sequence model
    pars = {'a': [0.03746, 0.3448, -0.1156],
            'a_err': [0.01739, 0.0297, 0.0107],
            'b': [0.9605, -0.04990, -0.05984],
            'b_err': [0.0100, 0.01518, 0.00482],
            'c': [0.2516, 1.118, -0.2006],
            'c_err': [0.0082, 0.013, 0.0039],
            'logMt': [10.22, 0.3826, -0.04491],
            'logMt_err': [0.01, 0.0188, 0.00613]
            }
    
    # Sample model parameters with random deviations
    logMt_sample = (np.random.normal(pars["logMt"][0], pars["logMt_err"][0]) + 
                    z0*np.random.normal(pars["logMt"][1], pars["logMt_err"][1]) + 
                    z0**2*np.random.normal(pars["logMt"][2], pars["logMt_err"][2]))
    
    a_sample = (np.random.normal(pars["a"][0], pars["a_err"][0]) + 
                z0*np.random.normal(pars["a"][1], pars["a_err"][1]) + 
                z0**2*np.random.normal(pars["a"][2], pars["a_err"][2]))
    
    b_sample = (np.random.normal(pars["b"][0], pars["b_err"][0]) + 
                z0*np.random.normal(pars["b"][1], pars["b_err"][1]) + 
                z0**2*np.random.normal(pars["b"][2], pars["b_err"][2]))
    
    c_sample = (np.random.normal(pars["c"][0], pars["c_err"][0]) + 
                z0*np.random.normal(pars["c"][1], pars["c_err"][1]) + 
                z0**2*np.random.normal(pars["c"][2], pars["c_err"][2]))

    # Calculate the center of the main sequence based on stellar mass
    if logM_sample > logMt_sample:
        logSFR_sample = a_sample*(logM_sample - logMt_sample) + c_sample
    else:
        logSFR_sample = b_sample*(logM_sample - logMt_sample) + c_sample

    return logSFR_sample



def generate_sfr(logM_sample, z0):
    """
    Generates a random sample of the star formation rate (SFR) for a galaxy
    at redshift z0 and stellar mass logM_sample.

    Parameters:
    ----------
    logM_sample   : float
        Logarithm of stellar mass sample.
    z0            : float
        Redshift.

    Returns:
    -------
    logSFR_sample : float
        Logarithm of randomly sampled star formation rate.
    """

    # Generate the center of the main sequence SFR
    logSFR_ms_center = generate_ms_center(logM_sample, z0)
    
    # Sample SFR from a normal distribution centered at the main sequence SFR
    logSFR_sample = np.random.normal(logSFR_ms_center, 0.35)
    
    return logSFR_sample



def sample_interp_sfr(sfrs, prob_density):
    """
    Samples an interpolated value of star formation rate (SFR) based on 
    provided probabilities.

    Parameters:
    ----------
    sfrs         : array_like
        Array of star formation rate values.
    prob_density : array_like
        Probability density function over sfrs, should sum to 1.

    Returns:
    -------
    interp_value : float
        Interpolated value of SFR.
    """
    
    # Normalize probability density
    prob_density /= prob_density.sum()
    
    # Randomly choose an interpolation factor between 0 and 1
    alpha_value = np.random.rand()
    
    # Choose an index based on the normalized probabilities
    index = np.random.choice(len(sfrs) - 1, p=prob_density[:-1])
    
    # Interpolate between neighboring SFR values
    interp_value = sfrs[index] * (1 - alpha_value) + sfrs[index + 1] * alpha_value
    
    return interp_value



def generate_logm_logsfr_z_pdf(ztarget):
    """
    Returns the density distribution of galaxies in the logM-logSFR plane at 
    redshift ztarget.

    Parameters:
    ----------
    ztarget : float
        Target redshift at which to generate the distribution.

    Returns:
    -------
    mgrid   : array_like
        Grid of stellar mass values.
    sfrgrid : array_like
        Grid of star formation rate values.
    density : array_like
        Density distribution of galaxies in the logM-logSFR plane at ztarget.
    """

    ndummy = 51
    zmin, zmax = ztarget - 0.01, ztarget + 0.01
    
    sfrgrid = np.linspace(sfrmin, sfrmax, nsfr)
    mgrid = np.linspace(mmin, mmax, nlogm)
    zgrid = np.arange(zmin, zmax + dz, dz)
    nz = zgrid.shape[0]
    dummy = np.linspace(-3., 3., ndummy)
    
    # Load trained Normalizing Flow
    flow = load_nf()
    
    # Compute probability density function
    prob_density = sample_density(nlogm, nsfr, nz, ndummy, dz,
                                  mgrid, sfrgrid, zgrid, dummy,
                                  flow, redshift_smoothing=True)

    # Select the index closest to ztarget
    zidx = np.abs(zgrid - ztarget).argmin()
    
    # Extract density at the selected redshift index
    density = prob_density[:, :, zidx]
    
    # Apply completeness cut based on redshift
    if ztarget > 0.2:
        below_mcomp = mgrid < threedhst_mass_completeness(ztarget)
    else:
        below_mcomp = mgrid < cosmos15_mass_completeness(ztarget)
    
    # Set density to zero below completeness limit
    density[below_mcomp, :] = 0

    return mgrid, sfrgrid, density.T



def generate_bkg_grid_below_mcomp():
    """
    Generate a grid of galaxy density distribution in logM-logSFR space over 
    a range of redshifts.

    Parameters:
    ----------
    None

    Returns:
    -------
    redshifts            : array_like
        Array of redshift values.
    bkg_grid_below_mcomp : list
        List of galaxy density distributions corresponding to each redshift.
    """
    redshifts = np.linspace(0.2, 1, 50)
    bkg_grid_below_mcomp = []
    
    for i in tqdm(range(len(redshifts))):
        z = redshifts[i]
        _, _, density = generate_logm_logsfr_z_pdf(z)
        bkg_grid_below_mcomp.append(density)

    return redshifts, bkg_grid_below_mcomp



def generate_samples_from_distribution(x, p, num_samples):
    """
    Returns num_samples from array x with probabilities p.

    Parameters:
    ----------
    x           : array_like
        Array of values to sample from.
    p           : array_like
        Probabilities corresponding to each value in x.
    num_samples : int
        Number of samples to generate.

    Returns:
    -------
    interpolated_samples : array_like
        Samples generated from the distribution defined by x and p.
    """
    p = p / np.sum(p)  # Normalize probabilities
    cdf = np.cumsum(p)  # Compute cumulative distribution function
    uniform_samples = np.random.rand(num_samples)  # Generate uniform samples
    interpolated_samples = np.interp(uniform_samples, cdf, x)  # Interpolate to get samples
    return interpolated_samples



def generate_bkg_z(ztarget, Nsamp, redshifts, bkg_grid_below_mcomp):
    """
    Generate Nsamp number of background galaxies logM and logSFR at redshift 
    ztarget.

    Parameters:
    ----------
    ztarget              : float
        Target redshift for generating background galaxies.
    Nsamp                : int
        Number of background galaxy samples to generate.
    redshifts            : array_like
        Array of redshift values.
    bkg_grid_below_mcomp : list
        List of galaxy density distributions corresponding to each redshift.

    Returns:
    -------
    logmstar_bkg         : array_like
        Array of logarithmic mass values for the background galaxies.
    logsfr_bkg           : array_like
        Array of logarithmic star formation rates for the background galaxies.
    """
    sfrgrid = np.linspace(sfrmin, sfrmax, nsfr)
    mgrid = np.linspace(mmin, mmax, nlogm)

    # Generate phi distribution at ztarget
    phi, _, _ = generate_phi(ztarget)
    logm = np.linspace(mmin, mmax, 100)
    
    # Generate background logM samples
    logmstar_bkg = generate_samples_from_distribution(logm, phi / np.sum(phi), Nsamp)
    
    logsfr_bkg = []
    
    if ztarget > 0.2:
        mcomp_limit = threedhst_mass_completeness(ztarget)
    else:
        zdummy = 0.25
        mcomp_limit = cosmos15_mass_completeness(zdummy)
    
    # Generate logsfr_bkg samples
    for k in range(Nsamp):
        logm_sample = logmstar_bkg[k]
        zidx = np.abs(redshifts - ztarget).argmin()
        prob_density = bkg_grid_below_mcomp[zidx].T
        
        if logm_sample > mcomp_limit + 0.3:
            # Sample from Leja et al. normalizing flows PDF
            logm_idx = np.arange(len(mgrid))[mgrid <= logm_sample][-1]
            prob_density = prob_density[logm_idx, :]
            
            ms_sfr_center_z = generate_ms_center(logm_sample, ztarget)
            ms_sfr_center_z0 = generate_ms_center(logm_sample, redshifts[zidx])
            offset_target = 0
            
            offset = ms_sfr_center_z - ms_sfr_center_z0
            logsfr_bkg.append(generate_samples_from_distribution(
                sfrgrid + offset - offset_target, prob_density, 1)[0])
        
        else:
            # Sample from extrapolated Leja et al.
            logm_idx = np.arange(len(mgrid))[mgrid <= (mcomp_limit + 0.5)][-1]
            prob_density = prob_density[logm_idx, :]
            
            ms_sfr_center_z = generate_ms_center((mcomp_limit + 0.5), 
                                                 redshifts[zidx])
            ms_sfr_center_z0 = generate_ms_center(logm_sample, ztarget)
            offset = ms_sfr_center_z0 - ms_sfr_center_z
            offset_target = 0
            
            logsfr_bkg.append(generate_samples_from_distribution(
                sfrgrid + offset - offset_target, prob_density, 1)[0])

    return logmstar_bkg, logsfr_bkg


