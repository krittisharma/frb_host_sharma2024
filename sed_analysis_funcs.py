#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Dec 7 10:00:00 2023

@author: kritti
"""

# set up some environmental dependencies
import os
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.pyplot import *
import seaborn as sns
import numpy as np, matplotlib.pyplot as plt
import warnings
from scipy.stats import truncnorm
os.environ['SPS_HOME'] = "/Users/krittisharma/Desktop/research/fsps"
from prospect.models import priors
from prospect.models.sedmodel import SedModel, PolySedModel
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['image.origin'] = 'lower'
sns.set_context('talk')
sns.set(font_scale=1.4)
sns.set_palette('colorblind')
sns.set_style('ticks')
plt.rcParams["font.family"] = "Times New Roman"


##### Mass-metallicity prior ######

class MassMet(priors.Prior):
    """
    A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt('galaxies_data/gallazzi_05_massmet.txt')

    def __len__(self):
        """ Hack to work with Prospector 0.3
        """
        return 2

    def scale(self, mass):
        """Calculate the scale parameter for the mass-metallicity 
        relationship."""
        upper_84 = np.interp(mass, self.massmet[:, 0], self.massmet[:, 3])
        lower_16 = np.interp(mass, self.massmet[:, 0], self.massmet[:, 2])
        return upper_84 - lower_16

    def loc(self, mass):
        """Calculate the location parameter for the mass-metallicity 
        relationship."""
        return np.interp(mass, self.massmet[:, 0], self.massmet[:, 1])

    def get_args(self, mass):
        """Calculate parameters 'a' and 'b' for the truncnorm 
        distribution."""
        a = (self.params['z_mini'] - self.loc(mass)) / self.scale(mass)
        b = (self.params['z_maxi'] - self.loc(mass)) / self.scale(mass)
        return [a, b]

    @property
    def range(self):
        """Return the range of prior parameters."""
        return ((self.params['mass_mini'], self.params['mass_maxi']),
                (self.params['z_mini'], self.params['z_maxi']))

    def bounds(self, **kwargs):
        """Return the bounds of the prior parameters."""
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the natural log of the prior probability at x."""
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.atleast_2d(np.zeros_like(x))
        a, b = self.get_args(x[..., 0])
        p[..., 1] = self.distribution.pdf(x[..., 1], a, b, 
                                          loc=self.loc(x[..., 0]), 
                                          scale=self.scale(x[..., 0]))
        with np.errstate(invalid='ignore'):
            p[..., 1] = np.log(p[..., 1])
        return p

    def sample(self, nsample=None, **kwargs):
        """Draw samples from the prior distribution."""
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = np.random.uniform(low=self.params['mass_mini'], 
                                 high=self.params['mass_maxi'], size=nsample)
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), 
                                    scale=self.scale(mass), size=nsample)
        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        """Transform values from CDF to parameter space."""
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = x[0] * (self.params['mass_maxi'] - self.params['mass_mini']) + \
            self.params['mass_mini']
        a, b = self.get_args(mass)
        met = self.distribution.ppf(x[1], a, b, loc=self.loc(mass), 
                                    scale=self.scale(mass))
        return np.array([mass, met])

    

##### Function ######
def normalize_ggc_spec(obs,
                       norm_band):
    """
    Normalize the spectrum to a photometric band.

    Parameters:
    ----------
    obs       : dict
        Dictionary containing observation data including wavelength, spectrum,
        filters, maggies, uncertainties, and optionally sky background.
    norm_band : str
        Name of the photometric band to which the spectrum will be normalized.

    Returns:
    -------
    obs       : dict
        Modified observation dictionary with the spectrum normalized to the
        specified band.
    """
    from sedpy.observate import getSED
    from prospect.sources.constants import lightspeed, jansky_cgs

    # Extract necessary information from observation dictionary
    bands = [f.name for f in obs['filters']]
    norm_index = bands.index(norm_band)

    # Calculate synthetic photometry
    synphot = getSED(obs['wavelength'], obs['spectrum'], obs['filters'])
    synphot = np.atleast_1d(synphot)

    # Compute normalization factor
    norm = 10**(-0.4 * synphot[norm_index]) / obs['maggies'][norm_index]

    # Convert spectrum to maggies and normalize
    wave = obs["wavelength"]
    flambda_to_maggies = wave * (wave/lightspeed) / jansky_cgs / 3631
    maggies = obs["spectrum"] / norm * flambda_to_maggies
    obs["spectrum"] = maggies

    # Normalize uncertainties
    obs["unc"] = obs["unc"] / norm * flambda_to_maggies

    # Normalize sky background if available
    if "sky" in obs:
        obs["sky"] = obs["sky"] / norm * flambda_to_maggies

    # Store the normalized band information
    obs["norm_band"] = norm_band

    return obs



def build_obs(filternames,
              mags_raw,
              mags_unc,
              corrections,
              phot_mask,
              wavelen=None,
              flux=None,
              flux_unc=None,
              mask=None,
              **extras):
    """
    Build a dictionary of observational data.

    Parameters:
    -----------
    filternames : list
        List of filter names.
    mags_raw    : array-like
        Array of raw magnitudes.
    mags_unc    : array-like
        Array of uncertainties in magnitudes.
    corrections : array-like
        Array of corrections to apply to raw magnitudes.
    phot_mask   : array-like
        Boolean mask indicating which observations are photometric.
    wavelen     : array-like, optional
        Wavelength array.
    flux        : array-like, optional
        Flux array.
    flux_unc    : array-like, optional
        Uncertainty in flux array.
    mask        : array-like, optional
        Mask array.
    **extras    : dict
        Additional keyword arguments.

    Returns:
    --------
    obs         : dict
        Dictionary containing observational data including filters, maggies,
        maggies_unc, phot_mask, phot_wave, wavelength, spectrum, unc, mask,
        and optionally normalized spectrum.
    """
    from prospect.utils.obsutils import fix_obs
    import sedpy

    obs = {}

    # Load filters
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # Apply corrections to magnitudes
    mags = mags_raw + corrections

    # Calculate maggies and uncertainties
    obs["maggies"] = 10**(-0.4 * mags)
    obs["maggies_unc"] = (((10**(0.4 * mags_unc)) - 1) * obs["maggies"])

    # Store photometric mask and effective wavelengths
    obs["phot_mask"] = phot_mask
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # Store optional data
    obs["wavelength"] = wavelen
    obs["spectrum"] = flux
    obs['unc'] = flux_unc
    obs['mask'] = mask

    # Optionally normalize spectrum
    if extras["normalize_spectrum"]:
        obs = normalize_ggc_spec(obs, norm_band=extras["norm_band"])

    # Optionally fix and return observation dictionary
    # return fix_obs(obs)
    return obs



def to_dust1(dust1_fraction=None, dust2=None, **extras):
    """
    Compute dust1 value based on dust1_fraction and dust2.

    Parameters:
    -----------
    dust1_fraction : float or None
        Fraction of dust1.
    dust2          : float or None
        Dust2 value.
    **extras       : dict
        Additional keyword arguments.

    Returns:
    --------
    dust1          : float or None
        Computed dust1 value, or None if inputs are None.
    """
    return dust1_fraction * dust2


    
def tage_rat(tage_tuniv, **extras):
    """
    Compute the age ratio based on tuniv and redshift.

    Parameters:
    -----------
    tage_tuniv : float
        Age relative to the universe age.
    **extras   : dict
        Additional keyword arguments.

    Returns:
    --------
    tage       : float
        Computed age ratio based on redshift and tuniv.
    """
    return tage_from_tuniv(extras["zred"], tage_tuniv)



def zred_to_agebins(zred=None,
                    nbins_sfh=None,
                    **extras):
    """
    Convert redshift to age bins based on SFH bins.

    Parameters:
    -----------
    zred      : float or array_like
        Redshift(s) of the object(s).
    nbins_sfh : int
        Number of SFH bins.
    **extras  : dict
        Additional keyword arguments.

    Returns:
    --------
    agebins   : numpy.ndarray
        Array of age bin edges.
    """
    tuniv = np.squeeze(cosmo.age(zred).to("yr").value)
    ncomp = np.squeeze(nbins_sfh)
    tbinmax = 0.9 * tuniv
    agelims = [0.0, 7.4772] + np.linspace(8.0, 
                                          np.log10(tbinmax), ncomp - 2).tolist(
                                          ) + [np.log10(tuniv)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T



def logmass_to_masses(logmass=None,
                      logsfr_ratios=None,
                      zred=None,
                      **extras):
    """
    Convert logarithmic mass to masses in different age bins.

    Parameters:
    -----------
    logmass       : float
        Logarithm of total stellar mass.
    logsfr_ratios : numpy.ndarray
        Logarithmic ratios of star formation rates in each age bin.
    zred          : float or array_like
        Redshift(s) of the object(s).
    **extras      : dict
        Additional keyword arguments.

    Returns:
    --------
    masses : numpy.ndarray
        Array of masses corresponding to each age bin.
    """
    agebins = zred_to_agebins(zred=zred, **extras)
    logsfr_ratios = np.clip(logsfr_ratios, -10, 10)
    nbins = agebins.shape[0]
    sratios = 10 ** logsfr_ratios
    dt = (10 ** agebins[:, 1] - 10 ** agebins[:, 0])
    coeffs = np.array([(1. / np.prod(sratios[:i])) *
                       (np.prod(dt[1:i+1]) / np.prod(dt[:i]))
                       for i in range(nbins)])
    m1 = (10 ** logmass) / coeffs.sum()
    return m1 * coeffs



def massmet_to_logmass(massmet=None, **extras):
    """
    Extract and return the stellar mass from mass-metallicity tuple.

    Parameters:
    -----------
    massmet : tuple
        Tuple containing (logMstar, logZsol).

    Returns:
    --------
    logmass : float
        Logarithm of stellar mass.
    """
    return massmet[0]


def massmet_to_logzsol(massmet=None, **extras):
    """
    Extract and return the stellar metallicity from mass-metallicity tuple.

    Parameters:
    -----------
    massmet : tuple
        Tuple containing (logMstar, logZsol).

    Returns:
    --------
    logzsol : float
        Logarithm of stellar metallicity (log Z/Z_sun).
    """
    return massmet[1]


def tie_gas_logz(logzsol=None, **extras):
    """
    Function to tie the gas metallicity to the stellar metallicity.

    Parameters:
    -----------
    logzsol : float
        Logarithm of stellar metallicity (log Z/Z_sun).

    Returns:
    --------
    logzsol : float
        Logarithm of gas metallicity (log Z_gas/Z_sun).
    """
    return logzsol



def build_sps(**extras):
    """
    Build and return an SPS (Stellar Population Synthesis) object based on 
    input parameters.

    Parameters:
    -----------
    use_parametric : bool
        Flag indicating whether to use parametric CSPSpecBasis (True) or 
        FastStepBasis (False).
    zcontinuous    : int
        If 1, ensures interpolation between SSPs for continuous metallicity 
        parameter (logzsol).

    Returns:
    --------
    sps : object
        Instance of CSPSpecBasis or FastStepBasis depending on use_parametric 
        flag.
    """
    if extras["use_parametric"]:
        from prospect.sources import CSPSpecBasis
        sps = CSPSpecBasis(zcontinuous=extras["zcontinuous"])
    else:
        from prospect.sources import FastStepBasis
        sps = FastStepBasis(zcontinuous=extras["zcontinuous"])
    return sps



def functions_to_names(p):
    """
    Replace prior and dust functions (or objects) with the names of those
    functions (or pickles).
    
    Parameters:
    -----------
    p : dict
        Dictionary containing functions or objects to be replaced.

    Returns:
    --------
    p : dict
        Updated dictionary where callables are replaced with names or pickled 
        representations.
    """
    import pickle

    # Iterate over items in dictionary
    for k, v in list(p.items()):
        # Check if the value is callable (function or object)
        if callable(v):
            try:
                # Try to get function name and module
                p[k] = [v.__name__, v.__module__]
            except AttributeError:
                # If it's not possible, pickle the object
                p[k] = pickle.dumps(v, protocol=2)

    return p



def flux_norm(self, **kwargs):
    """Compute the scaling required to go from Lsun/Hz/Msun to maggies.
    
    Note this includes the (1+z) factor required for flux densities.

    Parameters:
    -----------
    kwargs : dict
        Additional keyword arguments.
        Requires 'zred' for redshift information.

    Returns:
    --------
    norm   : float
        The normalization factor, scalar float.
    """
    if (kwargs["zred"] == 0) or ('lumdist' in self.params):
        lumdist = self.params.get('lumdist', 1e-5)
    else:
        from astropy.cosmology import WMAP9 as cosmo
        lumdist = cosmo.luminosity_distance(kwargs["zred"])
        lumdist = lumdist.to('Mpc').value

    dfactor = (lumdist * 1e5)**2
    mass = np.sum(self.params.get('mass', 1.0))
    
    from prospect.sources.constants import cosmo, jansky_cgs, to_cgs_at_10pc
    unit_conversion = to_cgs_at_10pc / (3631 * jansky_cgs) * \
                      (1 + kwargs["zred"])

    return mass * unit_conversion / dfactor



def nebline_photometry(self, filters, elams=None, elums=None):
    """Compute the emission line contribution to photometry.

    This requires several cached attributes:
      + ``_ewave_obs``
      + ``_eline_lum``

    Parameters:
    -----------
    filters : sedpy.observate.FilterSet or list
        Instance of sedpy.observate.FilterSet or a list of 
        sedpy.observate.Filter objects.
    elams   : array-like, optional
        The emission line wavelength in angstroms. If not supplied, 
        uses the cached `_ewave_obs` attribute.
    elums   : array-like, optional
        The emission line flux in erg/s/cm^2. If not supplied, uses 
        the cached `_eline_lum` attribute and applies appropriate distance 
        dimming and unit conversion.

    Returns:
    --------
    nebflux : ndarray
        The flux of the emission line through the filters, in units of maggies.
        ndarray of shape ``(len(filters),)``
    """
    if (elams is None) or (elums is None):
        elams = self._ewave_obs[self._use_eline]
        self.line_norm = self.flux_norm() / \
              (1 + self._zred) * (3631 * jansky_cgs)
        elums = self._eline_lum[self._use_eline] * self.line_norm

    flux = np.zeros(len(filters))
    try:
        flist = filters.filters
    except AttributeError:
        flist = filters
    
    for i, filt in enumerate(flist):
        trans = np.interp(elams, filt.wavelength, filt.transmission, 
                          left=0., right=0.)
        idx = (trans > 0)
        if True in idx:
            flux[i] = (trans[idx] * elams[idx] * elums[idx]).sum() / \
                filt.ab_zero_counts

    return flux



def absolute_rest_maggies(self, filters, **kwargs):
    """Return absolute rest-frame maggies (=10**(-0.4*M)) of the last
    computed spectrum.

    Parameters
    ----------
    filters : list of ``sedpy.observate.Filter()`` instances
        The filters through which you wish to compute the absolute mags

    Returns
    -------
    maggies : ndarray of shape (nbands,)
        The absolute restframe maggies of the model through the supplied
        filters, including emission lines.  Convert to absolute rest-frame
        magnitudes as M = -2.5 * log10(maggies)
    """
    spec, _, _ = self.mean_model(theta_max, obs, sps=sps)
    if kwargs["cal_spectrum"]:
        spec /= self._speccal

    from astropy.cosmology import WMAP9 as cosmo
    ld = cosmo.luminosity_distance(kwargs["zred"]).to("pc").value
    fmaggies = spec / (1 + kwargs["zred"]) * (ld / 10)**2

    from prospect.sources.constants import lightspeed, jansky_cgs
    from sedpy.observate import getSED

    flambda = fmaggies * lightspeed / wspec**2 * (3631 * jansky_cgs)
    abs_rest_maggies = 10**(-0.4 * np.atleast_1d(getSED(wspec, 
                                                        flambda, filters)))

    return abs_rest_maggies



def mi2mg(maggies):
    """Convert magnitudes to maggies.

    Parameters
    ----------
    maggies    : ndarray
        Array of magnitudes in maggies units.

    Returns
    -------
    magnitudes : ndarray
        Array of magnitudes corresponding to the input maggies.
    """
    return -2.5 * np.log10(maggies)



def build_model(**extras):
    """
    Build a model for the SED analysis.

    Parameters
    ----------
    extras : dict
        Extra parameters controlling model construction.

    Returns
    -------
    model : SedModel or PolySedModel
        An instance of the spectral energy distribution model.
    """

    # Selecting the type of SFH model
    model_params = TemplateLibrary["continuity_sfh"]

    # Redshift parameter
    model_params["zred"] = {
        "N": 1,
        "isfree": True,
        "init": extras["zred"],
        "units": "redshift",
        "prior": priors.TopHat(mini=extras["zred"] - 0.01, 
                               maxi=extras["zred"] + 0.01)
    }
    
    # Metallicity and mass parameters
    model_params['massmet'] = {
        'name': 'massmet',
        'N': 2,
        'isfree': True,
        "init": np.array([8.0, 0.0]),
        'prior': MassMet(z_mini=-2.00, z_maxi=0.19, mass_mini=8, mass_maxi=12)
    }

    model_params['logmass'] = {
        'N': 1,
        'isfree': False,
        'depends_on': massmet_to_logmass,
        'init': 10.0,
        'units': 'Msun',
        'prior': None
    }

    model_params['logzsol'] = {
        'N': 1,
        'isfree': False,
        'init': -0.5,
        'depends_on': massmet_to_logzsol,
        'units': r'$\log (Z/Z_\odot)$',
        'prior': None
    }

    # Number of age bins in SFH
    nbins = extras["nbins"]
    model_params["nbins_sfh"] = {
        'N': 1,
        'isfree': False,
        'init': nbins
    }

    # Age bins and SFH parameters
    model_params['agebins']['N'] = nbins
    model_params['agebins']['depends_on'] = zred_to_agebins
    model_params['mass']['N'] = nbins
    model_params['mass']['depends_on'] = logmass_to_masses

    model_params['logsfr_ratios'] = {
        'N': nbins - 1,
        'isfree': True,
        'init': np.full(nbins - 1, 0.0),
        'prior': priors.StudentT(mean=np.full(nbins - 1, 0.0),
                                 scale=np.full(nbins - 1, 0.3),
                                 df=np.full(nbins - 1, 2))
    }

    # Dust absorption parameters
    model_params['dust_type']['init'] = 4
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, 
                                                          mean=0.3, sigma=1)
    model_params["dust_index"] = {
        "N": 1,
        "isfree": True,
        "init": 0.0,
        "units": "power-law multiplication of Calzetti",
        "prior": priors.TopHat(mini=-1.0, maxi=0.4)
    }

    model_params['dust1'] = {
        "N": 1,
        "isfree": False,
        'depends_on': to_dust1,
        "init": 0.0,
        "units": "optical depth towards young stars",
        "prior": None
    }

    model_params['dust1_fraction'] = {
        'N': 1,
        'isfree': True,
        'init': 1.0,
        'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)
    }

    # Smooth the spectrum based on resolution
    if extras["smooth_spectrum"]:
        model_params.update(TemplateLibrary["spectral_smoothing"])
        model_params["sigma_smooth"] = {
            "prior": priors.TopHat(mini=50.0, maxi=500.0),
            "init": 200,
            "isfree": True
        }
    else:
        model_params.update(TemplateLibrary["spectral_smoothing"])
        model_params["sigma_smooth"] = {
            "init": 200,
            "isfree": False
        }

    # Add dust emission parameters if specified
    if extras["add_duste"]:
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_gamma']['init'] = 0.01
        model_params['duste_qpah']['prior'] = priors.TopHat(mini=0.5, maxi=7.0)
        model_params['duste_gamma']['isfree'] = False
        model_params['duste_umin']['isfree'] = False

    # Add AGN parameters if specified
    if extras["add_agn"]:
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['isfree'] = True
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

    # Add nebular emission parameters if specified
    if extras["add_nebular_emission"]:
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True
        _ = model_params["gas_logz"].pop("depends_on")
        model_params.update(TemplateLibrary['nebular_marginalization'])
        model_params['nebemlineinspec']['init'] = True
        model_params['eline_prior_width'] = {
            'N': 1,
            'isfree': True,
            'init': 1,
            'units': r'width of Gaussian prior on line luminosity, in units of \
                (true luminosity/FSPS predictions)',
            'prior': priors.TopHat(mini=0.01, maxi=100)
        }
        model_params['use_eline_prior']['init'] = True
        model_params["eline_sigma"] = {
            'N': 1,
            'isfree': True,
            'init': 200.0,
            'units': r'km/s',
            'prior': priors.TopHat(mini=30, maxi=500)
        }

    # Calibration and outlier models
    if extras["cal_spectrum"]:
        model_params.update(TemplateLibrary['optimize_speccal'])
        model_params["polyorder"]["init"] = extras["continuum_order"]
        model_params['spec_norm']['isfree'] = False
        model_params['spec_norm']['prior'] = priors.Normal(mean=1.0, sigma=0.3)
        model_params['f_outlier_spec'] = {
            "N": 1,
            "isfree": True,
            "init": 0.01,
            "prior": priors.TopHat(mini=1e-5, maxi=0.5)
        }
        model_params['nsigma_outlier_spec'] = {
            "N": 1,
            "isfree": False,
            "init": 50.0
        }
        model_params['f_outlier_phot'] = {
            "N": 1,
            "isfree": False,
            "init": 0.00,
            "prior": priors.TopHat(mini=0, maxi=0.5)
        }
        model_params['nsigma_outlier_phot'] = {
            "N": 1,
            "isfree": False,
            "init": 50.0
        }
        model_params['spec_jitter'] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "prior": priors.TopHat(mini=1.0, maxi=3.0)
        }

    # Instantiate the model object
    if extras["cal_spectrum"]:
        return PolySedModel(model_params)
    else:
        return SedModel(model_params)



def compute_quantile(x, q, weights=None):
    """
    Compute quantiles of a dataset x, with optional weights.

    Parameters
    ----------
    x : array_like
        Input array or object that can be converted to an array.
    q : array_like
        Quantile or sequence of quantiles to compute, should be in the range 
        [0, 1].
    weights : array_like, optional
        Array of weights of the same shape as x.

    Returns
    -------
    quantiles : ndarray
        Quantiles corresponding to the given q values.

    Notes
    -----
    This function computes weighted quantiles, which are useful when x 
    represents a distribution with uneven sampling or measurement errors.
    """

    # Convert q to percentiles
    percentiles = np.array(q) * 100.0

    if weights is None:
        return np.percentile(x, percentiles)
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        sorted_weights = weights[idx]
        cdf = np.cumsum(sorted_weights)
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()
