#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Dec 7 10:00:00 2023

@author: kritti
"""

# set up some environmental dependencies
import os
import argparse
import numpy as np
import pandas as pd
import astropy.cosmology.units as cu
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.pyplot import *
import seaborn as sns
import numpy as np, matplotlib.pyplot as plt
import warnings
os.environ['SPS_HOME'] = "/Users/krittisharma/Desktop/research/fsps"
import sedpy
from tqdm import tqdm
from prospect.fitting import lnprobfn, fit_model
from prospect.plotting.sfh import nonpar_recent_sfr, ratios_to_sfrs
import prospect.io.read_results as reader
import prospect.io.write_results as writer
warnings.filterwarnings("ignore")

from sed_analysis_funcs import *

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
        

def plot_obs(obs, run_params, xmin=3000, xmax=20000, ymin=1e-5, ymax=1e1):
    """
    Plot observational data including photometry, spectrum (if available),
    and filter transmission curves.

    Parameters
    ----------
    obs : dict
        Dictionary containing observational data:
        - 'maggies': Array of observed photometric fluxes (maggies).
        - 'maggies_unc': Array of uncertainties in photometric fluxes (maggies).
        - 'wavelength': Array of observed wavelengths (angstroms).
        - 'spectrum': Array of observed spectrum fluxes (maggies).
        - 'unc': Array of uncertainties in spectrum fluxes (maggies).
        - 'mask': Array indicating valid spectrum data points.
        - 'filters': List of filter objects with wavelength and transmission.

    run_params : dict
        Dictionary containing runtime parameters:
        - 'zred': Redshift of the source.
        - 'use_spectrum': Boolean flag indicating whether spectrum is included.

    xmin, xmax : float, optional
        Minimum and maximum wavelength for x-axis (angstroms).

    ymin, ymax : float, optional
        Minimum and maximum flux density for y-axis (maggies).

    Returns
    -------
    None
    """
    # Print keys of the obs dictionary for reference
    print("Obs Dictionary Keys:\n\n{}\n".format(obs.keys()))

    # Start plotting
    plt.figure(figsize=(8.5, 4.5))

    # Plot the photometry data
    plt.errorbar(obs['wphot'] / (1 + run_params["zred"]),
                 obs['maggies'],
                 yerr=obs['maggies_unc'],
                 fmt='o',
                 markersize=5,
                 label='Observed Photometry',
                 color='blue')

    if run_params["use_spectrum"]:
        # Plot the spectrum
        plt.plot(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                 obs['spectrum'][obs["mask"]],
                 label='Observed Spectrum',
                 alpha=0.8,
                 linestyle='solid',
                 lw=2,
                 color='red')
        plt.fill_between(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                         obs['spectrum'][obs["mask"]] - obs["unc"][obs["mask"]],
                         obs['spectrum'][obs["mask"]] + obs["unc"][obs["mask"]],
                         color="red",
                         alpha=0.4)

    # Plot filters
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t /= t.max()
        t *= ymax / ymin
        plt.loglog(w / (1 + run_params["zred"]), t * ymin,
                   lw=1, color='gray', alpha=0.7)

    # Labels and formatting
    plt.xlabel(r'$\lambda_{\mathrm{rest}}$ [$\AA$]')
    plt.ylabel('Flux Density [maggies]')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='upper right', fontsize=12)
    plt.tick_params(axis='both', direction='out', length=6, width=1)
    plt.minorticks_on()
    plt.tight_layout()

    # Save the plot
    plt.savefig("observed_data.png", dpi=400, bbox_inches='tight')

    # Show the plot (if needed)
    # plt.show()

    # Clear the figure
    plt.close()

    return None

    

def plot_initial_model(wspec, initial_spec, wphot, initial_phot, obs, run_params,
                       xmin=3000, xmax=20000, ymin=1e-5, ymax=1e1):
    """
    Plot the initial model spectrum and photometry alongside observed data.

    Parameters
    ----------
    wspec : ndarray
        Wavelength array for the model spectrum.

    initial_spec : ndarray
        Initial model spectrum flux densities.

    wphot : ndarray
        Wavelength array for the photometric filters.

    initial_phot : ndarray
        Initial model photometric flux densities.

    obs : dict
        Dictionary containing observational data:
        - 'maggies': Array of observed photometric fluxes (maggies).
        - 'maggies_unc': Array of uncertainties in photometric fluxes (maggies).
        - 'wavelength': Array of observed wavelengths (angstroms).
        - 'spectrum': Array of observed spectrum fluxes (maggies).
        - 'unc': Array of uncertainties in spectrum fluxes (maggies).
        - 'mask': Array indicating valid spectrum data points.
        - 'filters': List of filter objects with wavelength and transmission.

    run_params : dict
        Dictionary containing runtime parameters:
        - 'zred': Redshift of the source.
        - 'use_spectrum': Boolean flag indicating whether spectrum is included.

    xmin, xmax : float, optional
        Minimum and maximum wavelength for x-axis (angstroms).

    ymin, ymax : float, optional
        Minimum and maximum flux density for y-axis (maggies).

    Returns
    -------
    None
    """
    plt.figure(figsize=(8.5, 4.5))

    # Plot the model spectrum
    plt.loglog(wspec / (1 + run_params["zred"]),
               initial_spec,
               label='Model Spectrum',
               lw=0.7,
               color='navy',
               alpha=0.7)

    # Plot the initial model photometry
    plt.errorbar(wphot / (1 + run_params["zred"]),
                 initial_phot,
                 yerr=None,  # Assuming no uncertainties provided for initial model
                 label='Model Photometry',
                 marker='s',
                 markersize=10,
                 alpha=0.8,
                 ls='',
                 lw=2,
                 markerfacecolor='none',
                 markeredgecolor='blue',
                 markeredgewidth=1)

    # Plot the observed photometry
    plt.errorbar(wphot / (1 + run_params["zred"]),
                 obs['maggies'],
                 yerr=obs['maggies_unc'],
                 label='Observed Photometry',
                 marker='o',
                 markersize=10,
                 alpha=0.8,
                 ls='',
                 lw=2,
                 ecolor='red',
                 markerfacecolor='none',
                 markeredgecolor='red',
                 markeredgewidth=1)

    # Plot the observed spectrum if available
    if run_params["use_spectrum"]:
        plt.plot(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                 obs['spectrum'][obs["mask"]],
                 label='Observed Spectrum',
                 alpha=0.8,
                 linestyle='solid',
                 lw=2,
                 color='red')
        plt.fill_between(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                         obs['spectrum'][obs["mask"]] - obs["unc"][obs["mask"]],
                         obs['spectrum'][obs["mask"]] + obs["unc"][obs["mask"]],
                         color="red",
                         alpha=0.4)

    # Plot filters
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        plt.loglog(w / (1 + run_params["zred"]), t, lw=2, color='gray', alpha=0.7)

    # Axes labels and limits
    plt.xlabel(r'$\lambda_{\mathrm{rest}}$ [$\AA$]')
    plt.ylabel('Flux Density [maggies]')
    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='upper left', fontsize=12)
    plt.tick_params(axis='both', direction='out', length=6, width=1)
    plt.minorticks_on()
    plt.tight_layout()

    # Save the plot
    plt.savefig("initial_guess_model.png", dpi=400, bbox_inches='tight')

    # Clear the figure
    plt.close()

    return None
    

def plot_minimized_model(wspec, pspec, wphot, pphot, obs, run_params,
                         model, xmin=3000, xmax=20000, ymin=1e-5, ymax=1e1):
    """
    Plot the minimized model spectrum and photometry alongside observed data.

    Parameters
    ----------
    wspec : ndarray
        Wavelength array for the model spectrum.

    pspec : ndarray
        Minimized model spectrum flux densities.

    wphot : ndarray
        Wavelength array for the photometric filters.

    pphot : ndarray
        Minimized model photometric flux densities.

    obs : dict
        Dictionary containing observational data:
        - 'maggies': Array of observed photometric fluxes (maggies).
        - 'maggies_unc': Array of uncertainties in photometric fluxes (maggies).
        - 'wavelength': Array of observed wavelengths (angstroms).
        - 'spectrum': Array of observed spectrum fluxes (maggies).
        - 'unc': Array of uncertainties in spectrum fluxes (maggies).
        - 'mask': Array indicating valid spectrum data points.
        - 'filters': List of filter objects with wavelength and transmission.

    run_params : dict
        Dictionary containing runtime parameters:
        - 'zred': Redshift of the source.
        - 'cal_spectrum': Boolean flag indicating if spectrum is calibrated.
        - 'use_spectrum': Boolean flag indicating whether spectrum is included.

    model : object
        Object containing model information.

    xmin, xmax : float, optional
        Minimum and maximum wavelength for x-axis (angstroms).

    ymin, ymax : float, optional
        Minimum and maximum flux density for y-axis (maggies).

    Returns
    -------
    None
    """
    plt.figure(figsize=(8.5, 4.5))

    # Plot the model spectrum
    if run_params["cal_spectrum"]:
        plt.loglog(wspec / (1 + run_params["zred"]),
                   pspec / model._speccal,
                   label='Model Spectrum',
                   lw=2,
                   color='slateblue',
                   alpha=0.7)
    else:
        plt.loglog(wspec / (1 + run_params["zred"]),
                   pspec,
                   label='Model Spectrum',
                   lw=2,
                   color='slateblue',
                   alpha=0.7)

    # Plot the model photometry
    plt.errorbar(wphot / (1 + run_params["zred"]),
                 pphot,
                 label='Model Photometry',
                 marker='s',
                 markersize=10,
                 alpha=0.8,
                 ls='',
                 lw=2,
                 markerfacecolor='none',
                 markeredgecolor='slateblue',
                 markeredgewidth=2)

    # Plot the observed photometry
    plt.errorbar(wphot / (1 + run_params["zred"]),
                 obs['maggies'],
                 yerr=obs['maggies_unc'],
                 label='Observed Photometry',
                 marker='o',
                 markersize=10,
                 alpha=0.8,
                 ls='',
                 lw=2,
                 ecolor='green',
                 markerfacecolor='none',
                 markeredgecolor='green',
                 markeredgewidth=2)

    # Plot the observed spectrum if available
    if run_params["use_spectrum"]:
        if run_params["cal_spectrum"]:
            plt.plot(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                     obs['spectrum'][obs["mask"]] /
                     np.interp(obs["wavelength"][obs["mask"]],
                               wspec,
                               model._speccal),
                     label='Observed Spectrum',
                     alpha=0.8,
                     linestyle='solid',
                     lw=1,
                     color='red')
            plt.fill_between(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                             (obs['spectrum'][obs["mask"]] /
                              np.interp(obs["wavelength"][obs["mask"]],
                                        wspec,
                                        model._speccal) - obs["unc"][obs["mask"]]),
                             (obs['spectrum'][obs["mask"]] /
                              np.interp(obs["wavelength"][obs["mask"]],
                                        wspec,
                                        model._speccal) + obs["unc"][obs["mask"]]),
                             color="red",
                             alpha=0.4)
        else:
            plt.plot(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                     obs['spectrum'][obs["mask"]],
                     label='Observed Spectrum',
                     alpha=0.8,
                     linestyle='solid',
                     lw=1,
                     color='red')
            plt.fill_between(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                             obs['spectrum'][obs["mask"]] - obs["unc"][obs["mask"]],
                             obs['spectrum'][obs["mask"]] + obs["unc"][obs["mask"]],
                             color="red",
                             alpha=0.4)

    # Plot filter transmission curves
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2 * (np.log10(ymax / ymin))) * t * ymin
        plt.loglog(w / (1 + run_params["zred"]),
                   t,
                   lw=3,
                   color='gray',
                   alpha=0.7)

    # Axes labels and formatting
    plt.xlabel(r'$\lambda_{\mathrm{rest}}$ [$\AA$]')
    plt.ylabel('Flux Density [maggies]')
    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='upper left', fontsize=12, ncol=2)
    plt.tick_params(axis='both', direction='out', length=6, width=1)
    plt.minorticks_on()
    plt.tight_layout()

    # Save the plot
    plt.savefig("minimized_model.png", dpi=400, bbox_inches='tight')

    # Clear the figure
    plt.close()

    return None



def plot_minimized_zoomin_model(wspec, pspec, obs, run_params, model, theta_best,
                                xmin=3000, xmax=20000, ymin=1e-5, ymax=1e1):
    """
    Plot a zoomed-in view of the minimized model spectrum and observed spectrum.

    Parameters
    ----------
    wspec : ndarray
        Wavelength array for the model spectrum.

    pspec : ndarray
        Minimized model spectrum flux densities.

    obs : dict
        Dictionary containing observational data:
        - 'spectrum': Array of observed spectrum fluxes (maggies).
        - 'unc': Array of uncertainties in spectrum fluxes (maggies).
        - 'mask': Array indicating valid spectrum data points.
        - 'filters': List of filter objects with wavelength and transmission.

    run_params : dict
        Dictionary containing runtime parameters:
        - 'cal_spectrum': Boolean flag indicating if spectrum is calibrated.
        - 'use_spectrum': Boolean flag indicating whether spectrum is included.

    model : object
        Object containing model information.

    theta_best : float
        Best-fit parameter for redshift or other parameter used in scaling.

    xmin, xmax : float, optional
        Minimum and maximum wavelength for x-axis (angstroms).

    ymin, ymax : float, optional
        Minimum and maximum flux density for y-axis (maggies).

    Returns
    -------
    None
    """
    plt.figure(figsize=(8.5, 4.5))

    # Plot the model spectrum
    if run_params["cal_spectrum"]:
        plt.loglog(wspec / (1 + theta_best),
                   pspec / model._speccal,
                   label='Model Spectrum',
                   lw=2,
                   color='slateblue',
                   alpha=0.7)
    else:
        plt.loglog(wspec / (1 + theta_best),
                   pspec,
                   label='Model Spectrum',
                   lw=2,
                   color='slateblue',
                   alpha=0.7)

    # Plot the observed spectrum if available
    if run_params["use_spectrum"]:
        if run_params["cal_spectrum"]:
            plt.plot(obs["wavelength"][obs["mask"]] / (1 + theta_best),
                     obs['spectrum'][obs["mask"]] /
                     np.interp(obs["wavelength"][obs["mask"]],
                               wspec,
                               model._speccal),
                     label='Observed Spectrum',
                     alpha=0.8,
                     linestyle='solid',
                     lw=1,
                     color='red')
            plt.fill_between(obs["wavelength"][obs["mask"]] / (1 + theta_best),
                             (obs['spectrum'][obs["mask"]] /
                              np.interp(obs["wavelength"][obs["mask"]],
                                        wspec,
                                        model._speccal) - obs["unc"][obs["mask"]]),
                             (obs['spectrum'][obs["mask"]] /
                              np.interp(obs["wavelength"][obs["mask"]],
                                        wspec,
                                        model._speccal) + obs["unc"][obs["mask"]]),
                             color="red",
                             alpha=0.4)
        else:
            plt.plot(obs["wavelength"][obs["mask"]] / (1 + theta_best),
                     obs['spectrum'][obs["mask"]],
                     label='Observed Spectrum',
                     alpha=0.8,
                     linestyle='solid',
                     lw=1,
                     color='red')
            plt.fill_between(obs["wavelength"][obs["mask"]] / (1 + theta_best),
                             obs['spectrum'][obs["mask"]] - obs["unc"][obs["mask"]],
                             obs['spectrum'][obs["mask"]] + obs["unc"][obs["mask"]],
                             color="red",
                             alpha=0.4)

    # Axes labels and formatting
    plt.xlabel(r'$\lambda_{\mathrm{rest}}$ [$\AA$]')
    plt.ylabel('Flux Density [maggies]')
    plt.ylim([ymin, ymax])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='upper left', fontsize=12, ncol=2)
    plt.tick_params(axis='both', direction='out', length=6, width=1)
    plt.minorticks_on()
    plt.tight_layout()

    # Set x-axis limits based on observed data
    plt.xlim([np.min(obs["wavelength"][obs["mask"]]) / (1 + theta_best),
              np.max(obs["wavelength"][obs["mask"]]) / (1 + theta_best)])

    # Save the plot
    plt.savefig("minimized_zoomin_model.png", dpi=400, bbox_inches='tight')

    # Clear the figure
    plt.close()

    return None


    
def plot_sed_fit(wspec, pspec, pphot, obs, run_params, model, theta_max, frbs_db, frb,
                 xmin=3000, xmax=20000, ymin=1e-5, ymax=1e1):
    """
    Plot the spectral energy distribution (SED) fit including model spectrum, photometry,
    observed spectrum (if available), and residual plot.

    Parameters
    ----------
    wspec : ndarray
        Wavelength array for the model spectrum.

    pspec : ndarray
        Best-fit model spectrum flux densities.

    pphot : ndarray
        Best-fit model photometry flux densities.

    obs : dict
        Dictionary containing observational data:
        - 'spectrum': Array of observed spectrum fluxes (maggies).
        - 'maggies': Array of observed photometry fluxes (maggies).
        - 'maggies_unc': Array of uncertainties in photometry fluxes (maggies).
        - 'mask': Array indicating valid spectrum data points.
        - 'filters': List of filter objects with wavelength and transmission.

    run_params : dict
        Dictionary containing runtime parameters:
        - 'cal_spectrum': Boolean flag indicating if spectrum is calibrated.
        - 'use_spectrum': Boolean flag indicating whether spectrum is included.

    model : object
        Object containing model information.

    theta_max : float
        Maximum likelihood estimate of the model parameters.

    frbs_db : dict
        Database or source of FRB data.

    frb : str
        FRB identifier.

    xmin, xmax : float, optional
        Minimum and maximum wavelength for x-axis (angstroms).

    ymin, ymax : float, optional
        Minimum and maximum flux density for y-axis (maggies).

    Returns
    -------
    None
    """
    _, [ax, ax1] = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True,
                                 gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1})

    # Compute the best-fit model spectrum and photometry
    pspec_map, pphot_map, _ = model.mean_model(theta_max, obs=obs, sps=sps)

    # Plot model spectrum
    if run_params["cal_spectrum"]:
        ax.loglog(wspec / (1 + run_params["zred"]), pspec_map / model._speccal,
                  lw=0.7, color='green', alpha=0.7)
    else:
        ax.loglog(wspec / (1 + run_params["zred"]), pspec_map,
                  lw=0.7, color='green', alpha=0.7)

    # Plot model photometry
    ax.errorbar(wphot / (1 + run_params["zred"]), pphot_map,
                marker='s', markersize=10, alpha=0.8, ls='', lw=3,
                markerfacecolor='none', markeredgecolor='green', markeredgewidth=3)

    # Plot observed photometry
    ax.errorbar(wphot / (1 + run_params["zred"]), obs['maggies'],
                yerr=obs['maggies_unc'], ecolor='red', marker='o', markersize=10, ls='', lw=3,
                alpha=0.8, markerfacecolor='none', markeredgecolor='red', markeredgewidth=3)

    # Plot observed spectrum if available
    if run_params["use_spectrum"]:
        if run_params["cal_spectrum"]:
            ax.plot(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                    obs['spectrum'][obs["mask"]] / np.interp(wspec[obs["mask"]],
                                                            wspec,
                                                            model._speccal),
                    alpha=0.8, linestyle='solid', lw=1, color='red')
            ax.fill_between(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                            (obs['spectrum'][obs["mask"]] /
                             np.interp(wspec[obs["mask"]],
                                       wspec,
                                       model._speccal) - 0.05 * obs["unc"][obs["mask"]]),
                            (obs['spectrum'][obs["mask"]] /
                             np.interp(wspec[obs["mask"]],
                                       wspec,
                                       model._speccal) + 0.05 * obs["unc"][obs["mask"]]),
                            color="red", alpha=0.4)
        else:
            ax.plot(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                    obs['spectrum'][obs["mask"]],
                    alpha=0.8, linestyle='solid', lw=1, color='red')
            ax.fill_between(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                            (obs['spectrum'][obs["mask"]] - obs["unc"][obs["mask"]]),
                            (obs['spectrum'][obs["mask"]] + obs["unc"][obs["mask"]]),
                            color="red", alpha=0.4)

    # Plot filter transmission curves
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10 ** (0.2 * (np.log10(ymax / ymin))) * t * ymin
        ax.plot(w / (1 + run_params["zred"]), t, lw=2, color='gray', alpha=0.7)
        ax.set_yscale("log")

    ax.set_ylabel('Flux Density [maggies]')

    # Plot residual spectrum
    if run_params["use_spectrum"]:
        if run_params["cal_spectrum"]:
            ax1.plot(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                     -((pspec_map[obs["mask"]] /
                        model._speccal[obs["mask"]]) - (obs['spectrum'][obs["mask"]] /
                                                       np.interp(wspec[obs["mask"]],
                                                                 wspec,
                                                                 model._speccal))) /
                     (obs["unc"][obs["mask"]]),
                     lw=0.7, color='red', alpha=0.7)
        else:
            ax1.plot(obs["wavelength"][obs["mask"]] / (1 + run_params["zred"]),
                     -((pspec_map[obs["mask"]]) - (obs['spectrum'][obs["mask"]])) /
                     (obs["unc"][obs["mask"]]),
                     lw=0.7, color='red', alpha=0.7)

    # Plot residual photometry
    ax1.scatter(wphot / (1 + run_params["zred"]),
                -(pphot_map - obs['maggies']) / obs['maggies_unc'],
                marker='o', s=40, edgecolors="black", color="red")

    # Axis labels and formatting
    ax1.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\AA$]')
    ax1.set_ylabel(r'$\chi_{\mathrm{best}}$', labelpad=14, fontsize=19)
    ax1.axhline(0, linestyle="--", color="black", linewidth=1)

    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xscale("log")
    ax.set_xscale("log")
    ax1.set_xlim([xmin, xmax])
    ax.set_xlim([xmin, xmax])
    ax1.set_ylim([-10, 10])
    ax.set_ylim([ymin, ymax])

    # Text annotation
    ax.text(.05, .9, '{} ({})'.format(frbs_db["FRB_TNS_name"][0], frb),
            color="black", ha='left', va='center', transform=ax.transAxes,
            bbox=dict(facecolor='navajowhite', edgecolor='navajowhite', alpha=0.5))

    # Save the plot
    plt.savefig("{}_sed_fit.png".format(frb), dpi=400, bbox_inches='tight')

    return None


def plot_sed_zoomin_fit(wspec, pspec, obs, run_params, model, theta_max, frbs_db, frb,
                        xmin=3000, xmax=20000, ymin=1e-5, ymax=1e1):
    """
    Plot a zoomed-in view of the spectral energy distribution (SED) fit including
    observed spectrum (if available), model spectrum, and residual plot.

    Parameters
    ----------
    wspec : ndarray
        Wavelength array for the model spectrum.

    pspec : ndarray
        Best-fit model spectrum flux densities.

    obs : dict
        Dictionary containing observational data:
        - 'spectrum': Array of observed spectrum fluxes (maggies).
        - 'maggies_unc': Array of uncertainties in photometry fluxes (maggies).
        - 'mask': Array indicating valid spectrum data points.

    run_params : dict
        Dictionary containing runtime parameters:
        - 'cal_spectrum': Boolean flag indicating if spectrum is calibrated.
        - 'use_spectrum': Boolean flag indicating whether spectrum is included.

    model : object
        Object containing model information.

    theta_max : float
        Maximum likelihood estimate of the model parameters.

    frbs_db : dict
        Database or source of FRB data.

    frb : str
        FRB identifier.

    xmin, xmax : float, optional
        Minimum and maximum wavelength for x-axis (angstroms).

    ymin, ymax : float, optional
        Minimum and maximum flux density for y-axis (maggies).

    Returns
    -------
    None
    """
    _, [ax, ax1] = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True,
                                 gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1})

    # Compute the best-fit model spectrum
    pspec_map, _, _ = model.mean_model(theta_max, obs=obs, sps=sps)

    # Plot observed spectrum if available
    if run_params["use_spectrum"]:
        if run_params["cal_spectrum"]:
            ax.plot(obs["wavelength"][obs["mask"]] / (1 + theta_max[0]),
                    obs['spectrum'][obs["mask"]] / np.interp(wspec[obs["mask"]],
                                                            wspec,
                                                            model._speccal),
                    alpha=0.8, linestyle='solid', lw=1, color='red')
            ax.fill_between(obs["wavelength"][obs["mask"]] / (1 + theta_max[0]),
                            (obs['spectrum'][obs["mask"]] /
                             np.interp(wspec[obs["mask"]],
                                       wspec,
                                       model._speccal) - 0.05 * obs["unc"][obs["mask"]]),
                            (obs['spectrum'][obs["mask"]] /
                             np.interp(wspec[obs["mask"]],
                                       wspec,
                                       model._speccal) + 0.05 * obs["unc"][obs["mask"]]),
                            color="red", alpha=0.4)
        else:
            ax.plot(obs["wavelength"][obs["mask"]] / (1 + theta_max[0]),
                    obs['spectrum'][obs["mask"]],
                    alpha=0.8, linestyle='solid', lw=1, color='red')
            ax.fill_between(obs["wavelength"][obs["mask"]] / (1 + theta_max[0]),
                            (obs['spectrum'][obs["mask"]] - obs["unc"][obs["mask"]]),
                            (obs['spectrum'][obs["mask"]] + obs["unc"][obs["mask"]]),
                            color="red", alpha=0.4)

    # Plot residual spectrum
    if run_params["use_spectrum"]:
        if run_params["cal_spectrum"]:
            ax1.plot(obs["wavelength"][obs["mask"]] / (1 + theta_max[0]),
                     -((pspec_map[obs["mask"]] /
                        model._speccal[obs["mask"]]) - (obs['spectrum'][obs["mask"]] /
                                                       np.interp(wspec[obs["mask"]],
                                                                 wspec,
                                                                 model._speccal))) /
                     (obs["unc"][obs["mask"]]),
                     lw=0.7, color='red', alpha=0.7)
        else:
            ax1.plot(obs["wavelength"][obs["mask"]] / (1 + theta_max[0]),
                     -((pspec_map[obs["mask"]]) - (obs['spectrum'][obs["mask"]])) /
                     (obs["unc"][obs["mask"]]),
                     lw=0.7, color='red', alpha=0.7)

    # Plot model spectrum
    if run_params["cal_spectrum"]:
        ax.loglog(wspec / (1 + theta_max[0]),
                  pspec_map / model._speccal,
                  lw=1, color='black', alpha=1)
    else:
        ax.loglog(wspec / (1 + theta_max[0]),
                  pspec_map,
                  lw=1, color='black', alpha=1)

    # Axis labels and formatting
    ax.set_ylabel('Flux Density [maggies]')
    ax1.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\AA$]')
    ax1.set_ylabel(r'$\chi_{\mathrm{best}}$', labelpad=14, fontsize=19)
    ax1.axhline(0, linestyle="--", color="black", linewidth=1)

    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xscale("log")
    ax.set_xscale("log")
    ax1.set_xlim([np.min(obs["wavelength"][obs["mask"]]) / (1 + theta_max[0]),
                  np.max(obs["wavelength"][obs["mask"]]) / (1 + theta_max[0])])
    ax.set_xlim([np.min(obs["wavelength"][obs["mask"]]) / (1 + theta_max[0]),
                 np.max(obs["wavelength"][obs["mask"]]) / (1 + theta_max[0])])
    ax1.set_ylim([-10, 10])
    ax.set_ylim([ymin, ymax])

    # Text annotation
    ax.text(.05, .9, '{} ({})'.format(frbs_db["FRB_TNS_name"][0], frb),
            color="black", ha='left', va='center', transform=ax.transAxes,
            bbox=dict(facecolor='navajowhite', edgecolor='navajowhite', alpha=0.5))

    # Save the plot
    plt.savefig("{}_sed_zoomin_fit.png".format(frb), dpi=400, bbox_inches='tight')

    return None



def plot_sfh():
    """
    Plot the star formation history (SFH) from model parameters.
    """
    # Import necessary libraries
    params = model.params
    from prospect.plotting.corner import quantile

    # Set up the plot figure
    plt.figure(figsize=(6, 4.5))
    grid = plt.GridSpec(30, 10, wspace=0.5, hspace=20)
    ax1 = plt.subplot(grid[:, :])

    # Calculate lookback times and set up for SFH calculation
    lookback_times = 10**(model.params["agebins"] - 9)
    sfrs = []
    
    # Compute SFRs from model chain results
    for i in range(len(result["chain"])):
        sfrs.append(ratios_to_sfrs(result["chain"][i][8], 
                                   result["chain"][i][2:2+7-1], 
                                   agebins=model.params["agebins"]))
    sfrs = np.array(sfrs)
    
    ns = sfrs.shape[0]
    tvec = np.linspace(1e-3, lookback_times[-1][-1], 1000)
    eps = 0.01
    tlook = 10**model.params["agebins"] / 1e9 * np.array([1. + eps, 1. - eps])
    tlook = np.tile(tlook, (ns, 1, 1))
    tt = tlook.reshape(tlook.shape[0], -1)
    ss = np.array([sfrs, sfrs]).transpose(1, 2, 0).reshape(tlook.shape[0], -1)
    
    # Compute SFH by interpolating SFRs over time
    sfhs = np.array([np.interp(tvec, t, s, left=0, right=0) for t, s in zip(tt, ss)])
    weights = result.get('weights', None)
    
    # Compute quantiles and plot SFH
    sq = quantile(sfhs.T, q=[0.16, 0.50, 0.84], weights=weights)
    ax1.plot(tvec, sq[:, 1], lw=1, color='dodgerblue', label="50th Percentile Sample")
    ax1.fill_between(tvec, sq[:, 0], sq[:, 2], color='dodgerblue', alpha=0.3)

    # Set plot labels and scaling
    ax1.set_xlabel(r"$t_{\ell}$ [Gyr]")
    ax1.set_ylabel(r"SFR [M$_\odot$/yr]")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    t_gal = cosmo.age(run_params["zred"] * cu.redshift).value
    ax1.set_xlim(t_gal, 1e-2)

    # Add galaxy name annotation
    ax1.text(.05, .9, '{}'.format(frbs_db["gal_name"][0]), 
             color="black", ha='left', va='center', transform=ax1.transAxes,
             bbox=dict(facecolor='navajowhite', edgecolor='navajowhite', alpha=0.5))

    # Save the plot
    plt.savefig("{}_sfh.png".format(frb), dpi=400, bbox_inches='tight')


##################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--frb', type=str,
                    help="Name of the FRB you wanna play with?")
args = parser.parse_args()

frb = args.frb

os.chdir("frbs_data/dsa_results/{}".format(frb))

frbs_file = "frb_obs.csv"
use_parametric = "N"
use_spectrum = "Y"

frbs_db = pd.read_csv(frbs_file)

# store some meta-parameters that control the input arguments to this method
run_params = {}
run_params["zred"] = frbs_db["redshift"][0]

mags_raw = []
mags_unc = []
filternames = []

filterset = ['galex_FUV', 'galex_NUV',
             'ps1_g', 'ps1_r', 'ps1_i', 'ps1_z', 'ps1_y',
             'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',
             'wasp_g0', 'wasp_r0', 'wasp_i0', 'wasp_z0',
             'LRIS_V', 'LRIS_R', 'LRIS_G', 'DEIMOS_R',
             'twomass_J', 'twomass_H', 'twomass_Ks',
             'wirc_J', 'wirc_H', 'wirc_Ks',
             'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
             'decam_g', 'decam_r', 'decam_z']

for filt in filterset:
    if frbs_db[filt][0] != -99 and frbs_db[filt][0] != -100:
        filternames.append(filt)
        mags_raw.append(frbs_db[filt][0])
        mags_unc.append(float(frbs_db[filt+"_err"][0]))

if use_spectrum == "Y":
    run_params["use_spectrum"] = True
    run_params["continuum_order"] = 12
    
    if frbs_db["smooth_spectrum"][0] == "Y":
        run_params["smooth_spectrum"] = True
    else:
        run_params["smooth_spectrum"] = False
    
    if frbs_db["cal_spectrum"][0] == "Y":
        run_params["cal_spectrum"] = True
    else:
        run_params["cal_spectrum"] = False

    if frbs_db["normalize_spectrum"][0] == "Y":
        run_params["normalize_spectrum"] = True
    else:
        print("Normalization of spectrum not possible")
        run_params["normalize_spectrum"] = False

    run_params["norm_band"] = frbs_db["norm_band"][0]
    run_params["add_nebular_emission"] = True
else:
    run_params["use_spectrum"] = False
    run_params["continuum_order"] = 0
    run_params["smooth_spectrum"] = False
    run_params["cal_spectrum"] = False
    run_params["normalize_spectrum"] = False
    run_params["norm_band"] = "ps1_g"
    run_params["add_nebular_emission"] = False

if use_parametric == "Y":
    run_params["use_parametric"] = True
    label = "parametric"
else:
    run_params["use_parametric"] = False
    label = "nonparametric"

run_params["nbins"] = 7

if frbs_db["add_duste"][0] == "Y":
    run_params["add_duste"] = True
else:
    run_params["add_duste"] = False

if frbs_db["add_agn"][0] == "Y":
    run_params["add_agn"] = True
else:
    run_params["add_agn"] = False

# run_params["add_burst"] = False
run_params["zcontinuous"] = 1
run_params["nwalkers"] = 128
run_params["niter"] = 1024
run_params["nburn"] = [32, 64, 128]
run_params["ldist"] = cosmo.luminosity_distance(run_params["zred"]).value
# luminosity distance to convert apparent magnitudes to observed fluxes
results_type = "dynesty"

xmin, xmax = frbs_db["xmin"][0], frbs_db["xmax"][0]
ymin, ymax = frbs_db["ymin"][0], frbs_db["ymax"][0]

if run_params["use_spectrum"]:
    spectrum_file = frbs_db["spectrum_file"][0]
outfile = "{}.h5".format(frbs_db["frb_name"][0])

# Magnitudes feeded into this pipeline are already corrected for dust
# extinction and AB magnitude system corrections
corrections = np.zeros(np.shape(mags_raw))
phot_mask = [True for i in range(len(mags_raw))]

if run_params["use_spectrum"]:
    if ".spec" in spectrum_file:
        with open(spectrum_file, 'r') as fin:
            data=[]
            for line in fin:
                words = line.split()
                if "#" in words:
                    continue
                elif "##" in words:
                    continue
                else:
                    data.append([float(word) for word in words])
        data = pd.DataFrame(data,
                            columns=["wavelength",  # Angstroms
                                    "flux",  # erg/cm^2/s/Ang
                                    "sky_flux",  # erg/cm^2/s/Ang
                                    "flux_unc",  # erg/cm^2/s/Ang
                                    "xpixel",
                                    "ypixel",
                                    "response",
                                    "flag"]
                            )
        data = data[data["wavelength"]/(1+run_params["zred"]) > 3200].reset_index()
        data = data[data["wavelength"]/(1+run_params["zred"]) < 5200]
        print(data.head())
        
    elif ".dat" in spectrum_file:
        with open(spectrum_file, 'r') as fin:
            data=[]
            for line in fin:
                words = line.split()
                if "#" in words:
                    continue
                elif "##" in words:
                    continue
                else:
                    data.append([float(word) for word in words])
        data = pd.DataFrame(data,
                            columns = ["wavelen", # Angstroms
                                       "flux"]
                           )
        data["flux"]= data["flux"]*(10**-23)/(data["wavelen"]*(data["wavelen"]*10**-8)/(3*10**10))
        data["flux_unc"] = 0.1*data["flux"]
        data["wavelength"] = data["wavelen"]

    else:
        data = pd.read_csv(spectrum_file)
        data["wavelength"] = data["wavelen"]
        data["flux"] = frbs_db["flux_norm"][0]*data["flux"]
        print("Normalized spectrum manually by a factor of {}".format(str(frbs_db["flux_norm"][0])))
        if frbs_db["flux_norm"][0] < 1:
            data["flux_unc"] = frbs_db["flux_norm"][0]*data["flux_unc"]
        else:
            data["flux_unc"] = 0.1*data["flux_unc"]

    print(data.head())
    resolution = 0.8
    wavelen_below = np.arange(xmin, data["wavelength"][0], resolution) 
    wavelen_above = np.arange(list(data["wavelength"])[-1], xmax*(1+run_params["zred"]), resolution)
    wavelens = np.array(list(wavelen_below) +
        list(data["wavelength"])+list(wavelen_above))
    
    fluxes_below = np.zeros(np.shape(wavelen_below))
    fluxes_above = np.zeros(np.shape(wavelen_above))
    fluxes = np.array(list(fluxes_below) +
        list(data["flux"])+list(fluxes_above))

    flux_uncs_below = np.zeros(np.shape(wavelen_below))
    flux_uncs_above = np.zeros(np.shape(wavelen_above))
    flux_uncs = np.array(list(flux_uncs_below) +
        list(data["flux_unc"])+list(flux_uncs_above))

    mask_below = [False for i in range(len(wavelen_below))]
    mask_mid = [True for i in range(len(data))] # list(np.logical_and(data["flux"]/data["flux_unc"]>5, data["flux"]/data["flux_unc"]<40))
    mask_above = [False for i in range(len(wavelen_above))]
    mask = mask_below+mask_mid+mask_above

    data = pd.DataFrame({"wavelen": wavelens,
                        "flux": fluxes,
                        "flux_unc": flux_uncs,
                        "mask": mask})

    wavelen = np.asarray(data["wavelen"], dtype="float")
    flux = np.asarray(data["flux"]*(10**23)*data["wavelen"] *
                      (data["wavelen"]*10**-8)/((3*10**10)*3631),
                      dtype="float")
    flux_unc = np.asarray(data["flux_unc"]*(10**23)*data["wavelen"] *
                          (data["wavelen"]*10**-8)/((3*10**10)*3631),
                          dtype="float")
    mask = data["mask"]

    # flux = convolve(flux, Box1DKernel(50))
    # flux_unc = convolve(flux_unc, Box1DKernel(50))
else:
    wavelen = None
    flux = None
    flux_unc = None
    mask = None

obs = build_obs(filternames=filternames,
                mags_raw=np.array(mags_raw),
                mags_unc=np.array(mags_unc),
                corrections=corrections,
                phot_mask=phot_mask,
                wavelen=wavelen,
                flux=flux,
                flux_unc=flux_unc,
                mask=mask,
                **run_params)

# Plot all the data at hand
wphot = obs["phot_wave"]
plot_obs()

model = build_model(**run_params)
print(model)
print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))
print("Initial parameter dictionary:\n{}".format(model.params))

sps = build_sps(**run_params)

# Generate the model SED at the initial value of theta
theta = model.theta.copy()

initial_spec, initial_phot, initial_mfrac = model.sed(theta,
                                                      obs=obs,
                                                      sps=sps)

title_text = ','.join(["{}={}".format(p, model.params[p][0])
                    for p in model.free_params])
print("Initial guess for the model parameters:")
print(title_text)

a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
# spectroscopic wavelengths
if obs["wavelength"] is None:
    wspec = sps.wavelengths
    wspec *= a  # redshift them
else:
    wspec = obs["wavelength"]

plot_initial_model()

# Set up MPI. Note that only model evaluation is parallelizable in dynesty,
# and many operations (e.g. new point proposal) are still done in serial.
# This means that single-core fits will always be more efficient for large
# samples. having a large ratio of (live points / processors) helps efficiency
# Scaling is: S = K ln(1 + M/K), where M = number of processes and K = number of live points
# Run as: mpirun -np <number of processors> python demo_mpi_params.py
# try:
#     import mpi4py
#     from mpi4py import MPI
#     from schwimmbad import MPIPool

#     mpi4py.rc.threads = False
#     mpi4py.rc.recv_mprobe = False

#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()

#     withmpi = comm.Get_size() > 1
# except ImportError:
#     print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
#     withmpi = False

try:
    result, _, _ = reader.results_from(outfile, dangerous=False)
except:
    # --- start minimization ----
    run_params["dynesty"] = False
    run_params["emcee"] = False
    run_params["optimize"] = True
    run_params["min_method"] = 'lm'
    run_params["nmin"] = 64

    # ensure that each processor runs its own version of FSPS
    # this ensures no cross-over memory usage
    from prospect.fitting import lnprobfn
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    # with MPIPool() as pool:
    #         # The subprocesses will run up to this point in the code
    #         if not pool.is_master():
    #             pool.wait()
    #             sys.exit(0)
    #         nprocs = pool.size
    #         output = fit_model(obs, model, sps, 
    #                            pool=pool, queue_size=nprocs, 
    #                            lnprobfn=lnprobfn_fixed, **run_params)

    output = fit_model(obs,
                    model,
                    sps,
                    lnprobfn=lnprobfn,
                    **run_params)
    print("Done optmization in {}s".format(output["optimization"][1]))
    (results, topt) = output["optimization"]

    # Find which of the minimizations gave the best result,
    # and use the parameter vector for that minimization
    ind_best = np.argmin([r.cost for r in results])
    theta_best = results[ind_best].x.copy()
    print(theta_best)
    title_text = ','.join(["{}={}".format(p, model.params[p])
                        for p in model.free_params])
    print(title_text)
    # generate model
    pspec, pphot, pfrac = model.mean_model(theta_best,
                                        obs=obs,
                                        sps=sps)

    plot_minimized_model()
    plot_minimized_zoomin_model()

    from prospect.fitting import lnprobfn
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    run_params["optimize"] = False
    run_params["emcee"] = False
    run_params["dynesty"] = True

    # add in dynesty settings
    run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
    run_params['nested_nlive_batch'] = 200 # 500
    run_params['nested_walks'] = 48  # sampling gets very inefficient w/ high S/N spectra
    run_params['nested_nlive_init'] = 400 # 500 
    run_params['nested_dlogz_init'] = 0.05
    run_params['nested_maxcall'] = 7500000
    run_params['nested_maxcall_init'] = 7500000
    run_params['nested_sample'] = 'rwalk'
    run_params['nested_maxbatch'] = None
    run_params['nested_posterior_thresh'] = 0.05 # 0.03
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}
    run_params['nested_target_n_effective'] = 1000
    
    output = fit_model(obs, model, sps, lnprobfn=lnprobfn_fixed, **run_params)
    print('done dynesty in {0}s'.format(output["sampling"][1]))

    writer.write_hdf5(outfile,
                    run_params,
                    model,
                    {"maggies": obs["maggies"]},
                    output["sampling"][0],
                    None,
                    tsample=output["sampling"][1],
                    toptimize=output["optimization"][1]) 

    result, _, _ = reader.results_from(outfile, dangerous=False)

weights = result.get('weights', None)

imax = np.argmax(result['lnprobability'])
if results_type == "emcee":
    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()
    thin = 6
else:
    theta_max = result["chain"][imax, :]
    thin = 1

print('MAP value: {}'.format(theta_max))

pspec_map, pphot_map, pfrac_map = model.mean_model(theta_max,
                                                obs=obs,
                                                sps=sps)

data = pd.DataFrame({"wavelen": wavelens,
                     "flux": fluxes, 
                     "flux_unc": flux_uncs,
                     "mask": mask})
flux = np.asarray(data["flux"]*(10**23)*data["wavelen"] *
                      (data["wavelen"]*10**-8)/((3*10**10)*3631),
                      dtype="float")
if run_params["cal_spectrum"]:
    flux[mask] = obs['spectrum'][mask]/np.interp(obs["wavelength"][mask],
                                wspec,
                                model._speccal)
data["flux"] = np.asarray(flux*(10**-23)*(data["wavelen"]**-1)*
                  ((data["wavelen"]**-1)*10**8)*((3*10**10)*3631),
                  dtype="float")
data["flux_unc"] = np.asarray(flux_unc*(10**-23)*(data["wavelen"]**-1)*
                      ((data["wavelen"]**-1)*10**8)*((3*10**10)*3631),
                      dtype="float")
data = data[data["mask"]==True]
data.to_csv("spec_calibrated.csv")


plot_sed_fit()

plot_sed_zoomin_fit()  

plot_sfh()

cornerfig = reader.subcorner(result, start=0, thin=thin, truths=theta_max, 
                            fig=subplots(len(model.free_params)+5+1,
                                         len(model.free_params)+5+1,
                                         figsize=(26,26))[0])
plt.savefig("{}_all_params.png".format(frb),
            dpi=400,
            bbox_inches = 'tight')

Mstars = []
Mstars_wts = []
for i in tqdm(range(100)):
    j = np.random.randint(len(result["chain"]))
    theta = result['chain'][j]
    mspec, mphot, mextra = model.mean_model(theta,
                                            obs,
                                            sps=sps)
    Mstars.append(np.log10(10**(theta[8])*mextra))
    Mstars_wts.append(weights[j])
Mstars = np.array(Mstars)
Mstars_wts = np.array(Mstars_wts)

frbs_db["logMstar"][0] = compute_quantile(Mstars, [0.5], Mstars_wts)[0]
frbs_db["logMstar_errl"][0] = compute_quantile(Mstars, [0.16], Mstars_wts)[0] - compute_quantile(Mstars, [0.5], Mstars_wts)[0]
frbs_db["logMstar_erru"][0] = compute_quantile(Mstars, [0.84], Mstars_wts)[0] - compute_quantile(Mstars, [0.5], Mstars_wts)[0]

sfr_20Myr = []
sfr_100Myr = []
ssfrs = []
params = model.params

for i in tqdm(range(len(result["chain"]))):
    sfr = nonpar_recent_sfr(np.array([result["chain"][i, 8]]), 
                            np.array([result["chain"][i, 2:2+7-1]]), 
                            np.array(params["agebins"]), 
                            sfr_period=0.02)
    sfr_20Myr.append(sfr)
    sfr = nonpar_recent_sfr(np.array([result["chain"][i, 8]]), 
                            np.array([result["chain"][i, 2:2+7-1]]), 
                            np.array(params["agebins"]))
    sfr_100Myr.append(sfr)
    mass = result["chain"][i, 8]
    ssfr = sfr*1e9/(10**mass)
    ssfrs.append(ssfr)

sfr_20Myr = np.array(sfr_20Myr)[:, 0]
sfr_100Myr = np.array(sfr_100Myr)[:, 0]
ssfrs = np.array(ssfrs)[:, 0]

frbs_db["sfr_20Myr"][0] = compute_quantile(sfr_20Myr, [0.5], weights)[0]
frbs_db["sfr_20Myr_errl"][0] = compute_quantile(sfr_20Myr, [0.16], weights)[0] - compute_quantile(sfr_20Myr, [0.5], weights)[0]
frbs_db["sfr_20Myr_erru"][0] = compute_quantile(sfr_20Myr, [0.84], weights)[0] - compute_quantile(sfr_20Myr, [0.5], weights)[0]

frbs_db["sfr_100Myr"][0] = compute_quantile(sfr_100Myr, [0.5], weights)[0]
frbs_db["sfr_100Myr_errl"][0] = compute_quantile(sfr_100Myr, [0.16], weights)[0] - compute_quantile(sfr_100Myr, [0.5], weights)[0]
frbs_db["sfr_100Myr_erru"][0] = compute_quantile(sfr_100Myr, [0.84], weights)[0] - compute_quantile(sfr_100Myr, [0.5], weights)[0]

frbs_db["ssfr"][0] = compute_quantile(ssfrs, [0.5], weights)[0]
frbs_db["ssfr_errl"][0] = compute_quantile(ssfrs, [0.16], weights)[0] - compute_quantile(ssfrs, [0.5], weights)[0]
frbs_db["ssfr_erru"][0] = compute_quantile(ssfrs, [0.84], weights)[0] - compute_quantile(ssfrs, [0.5], weights)[0]

dust1 = []
logmass = []
logzsol = []

for i in tqdm(range(len(result["chain"]))):
    dust1.append(result["chain"][i, 6+np.where(np.array(model.free_params) == "dust1_fraction")[0][0]]*result["chain"][i, np.where(np.array(model.free_params) == "dust2")[0][0]])
    logmass.append(result["chain"][i, 8])
    logzsol.append(result["chain"][i, 9])
    
frbs_db["duste_young"][0] = compute_quantile(dust1, [0.5], weights)[0]
frbs_db["duste_young_errl"][0] = compute_quantile(dust1, [0.16], weights)[0] - compute_quantile(dust1, [0.5], weights)[0]
frbs_db["duste_young_erru"][0] = compute_quantile(dust1, [0.84], weights)[0] - compute_quantile(dust1, [0.5], weights)[0]

frbs_db["logmass"][0] = compute_quantile(logmass, [0.5], weights)[0]
frbs_db["logmass_errl"][0] = compute_quantile(logmass, [0.16], weights)[0] - compute_quantile(logmass, [0.5], weights)[0]
frbs_db["logmass_erru"][0] = compute_quantile(logmass, [0.84], weights)[0] - compute_quantile(logmass, [0.5], weights)[0]

frbs_db["logzsol"][0] = compute_quantile(logzsol, [0.5], weights)[0]
frbs_db["logzsol_errl"][0] = compute_quantile(logzsol, [0.16], weights)[0] - compute_quantile(logzsol, [0.5], weights)[0]
frbs_db["logzsol_erru"][0] = compute_quantile(logzsol, [0.84], weights)[0] - compute_quantile(logzsol, [0.5], weights)[0]

for param in model.free_params:
    if param in ["logsfr_ratios", 'eline_prior_width', 'eline_sigma', 'massmet',
                 'sigma_smooth', 'gas_logz', 'f_outlier_spec', 'spec_jitter']:
        continue
    elif param in ['zred', 'dust2']:
        param_model = []
        for i in range(len(result["chain"])):
            param_model.append(result["chain"][i, np.where(np.array(model.free_params) == param)[0][0]])
        frbs_db[param][0] = compute_quantile(param_model, [0.5], weights)[0]
        frbs_db["{}_errl".format(param)][0] = compute_quantile(param_model, [0.16], weights)[0] - compute_quantile(param_model, [0.5], weights)[0]
        frbs_db["{}_erru".format(param)][0] = compute_quantile(param_model, [0.84], weights)[0] - compute_quantile(param_model, [0.5], weights)[0]
    else:
        param_model = []
        for i in range(len(result["chain"])):
            param_model.append(result["chain"][i, 6+np.where(np.array(model.free_params) == param)[0][0]])
        frbs_db[param][0] = compute_quantile(param_model, [0.5], weights)[0]
        frbs_db["{}_errl".format(param)][0] = compute_quantile(param_model, [0.16], weights)[0] - compute_quantile(param_model, [0.5], weights)[0]
        frbs_db["{}_erru".format(param)][0] = compute_quantile(param_model, [0.84], weights)[0] - compute_quantile(param_model, [0.5], weights)[0]

M_us, M_gs, M_rs, M_Vs = [], [], [], []
M_mag_wts = []
filters = sedpy.observate.load_filters(['sdss_u0',
                                        'sdss_g0',
                                        'sdss_r0',
                                        'subaru_suprimecam_V'])

for k in tqdm(range(100)):
    i = np.random.randint(len(result["chain"]))
    M_mag_wts.append(weights[i])
    spec, _, _ = model.mean_model(result['chain'][i], obs, sps=sps)
    spec/=model._speccal
    from astropy.cosmology import WMAP9 as cosmo
    ld = cosmo.luminosity_distance(run_params["zred"]).to("pc").value
    fmaggies = spec / (1 + run_params["zred"]) * (ld / 10)**2
    from prospect.sources.constants import cosmo, lightspeed, ckms, jansky_cgs
    flambda = fmaggies * lightspeed / wspec**2 * (3631*jansky_cgs)
    from sedpy.observate import getSED
    M_u, M_g, M_r, M_V = 10**(-0.4 * np.atleast_1d(getSED(wspec,
                                                           flambda,
                                                           filters)))
    M_us.append(M_u)
    M_gs.append(M_g)
    M_rs.append(M_r)
    M_Vs.append(M_V)

M_us = np.array(M_us)
M_gs = np.array(M_gs)
M_rs = np.array(M_rs)
M_Vs = np.array(M_Vs)
M_mag_wts = np.array(M_mag_wts)

frbs_db["M_r"][0] =  compute_quantile(np.array(mi2mg(M_rs)), [0.5], M_mag_wts)[0]
frbs_db["M_r_errl"][0] =  compute_quantile(np.array(mi2mg(M_rs)), [0.16], M_mag_wts)[0] - compute_quantile(np.array(mi2mg(M_rs)), [0.5], M_mag_wts)[0]
frbs_db["M_r_erru"][0] =  compute_quantile(np.array(mi2mg(M_rs)), [0.84], M_mag_wts)[0] - compute_quantile(np.array(mi2mg(M_rs)), [0.5], M_mag_wts)[0]

frbs_db["g-r"][0] =  compute_quantile(np.array(mi2mg(np.array(M_gs))-mi2mg(np.array(M_rs))), [0.5], M_mag_wts)[0]
frbs_db["g-r_errl"][0] =  compute_quantile(np.array(mi2mg(np.array(M_gs))-mi2mg(np.array(M_rs))), [0.16], M_mag_wts)[0] - compute_quantile(np.array(mi2mg(np.array(M_gs))-mi2mg(np.array(M_rs))), [0.5], M_mag_wts)[0]
frbs_db["g-r_erru"][0] =  compute_quantile(np.array(mi2mg(np.array(M_gs))-mi2mg(np.array(M_rs))), [0.84], M_mag_wts)[0] - compute_quantile(np.array(mi2mg(np.array(M_gs))-mi2mg(np.array(M_rs))), [0.5], M_mag_wts)[0]

frbs_db["u-r"][0] =  compute_quantile(np.array(mi2mg(np.array(M_us))-mi2mg(np.array(M_rs))), [0.5], M_mag_wts)[0]
frbs_db["u-r_errl"][0] =  compute_quantile(np.array(mi2mg(np.array(M_us))-mi2mg(np.array(M_rs))), [0.16], M_mag_wts)[0] - compute_quantile(np.array(mi2mg(np.array(M_us))-mi2mg(np.array(M_rs))), [0.5], M_mag_wts)[0]
frbs_db["u-r_erru"][0] =  compute_quantile(np.array(mi2mg(np.array(M_us))-mi2mg(np.array(M_rs))), [0.84], M_mag_wts)[0] - compute_quantile(np.array(mi2mg(np.array(M_us))-mi2mg(np.array(M_rs))), [0.5], M_mag_wts)[0]

lum = np.array((10**(np.array(mi2mg(M_Vs))/-2.5)*3.0128*(10**28)*10**7)/(3.839e33))
luminosities = [np.log10(lum[i]) for i in range(len(lum))]

frbs_db["log(nu*l_nu)"][0] =  compute_quantile(np.array(luminosities), [0.5], M_mag_wts)[0]
frbs_db["log(nu*l_nu)_errl"][0] =  compute_quantile(np.array(luminosities), [0.16], M_mag_wts)[0] - compute_quantile(np.array(luminosities), [0.5], M_mag_wts)[0]
frbs_db["log(nu*l_nu)_erru"][0] =  compute_quantile(np.array(luminosities), [0.84], M_mag_wts)[0] - compute_quantile(np.array(luminosities), [0.5], M_mag_wts)[0]

lookback_times = 10**(model.params["agebins"]-9)
sfrs = []
masses = []
for i in range(len(result["chain"])):
    # i = np.random.randint(len(result["chain"]))
    sfrs.append(ratios_to_sfrs(result["chain"][i][8], 
                               result["chain"][i][2:2+7-1], 
                               agebins=model.params["agebins"]))
    masses.append(result["chain"][i][8])
sfrs = np.array(sfrs)
masses = np.array(masses)

tbins = np.array([-lookback_times[0, 0]+lookback_times[0, 1],
         -lookback_times[1, 0]+lookback_times[1, 1],
         -lookback_times[2, 0]+lookback_times[2, 1],
         -lookback_times[3, 0]+lookback_times[3, 1],
         -lookback_times[4, 0]+lookback_times[4, 1],
         -lookback_times[5, 0]+lookback_times[5, 1],
         -lookback_times[6, 0]+lookback_times[6, 1]])*1e9
tbins_age = np.array([(lookback_times[0, 0]+lookback_times[0, 1])/2,
         (lookback_times[1, 0]+lookback_times[1, 1])/2,
         (lookback_times[2, 0]+lookback_times[2, 1])/2,
         (lookback_times[3, 0]+lookback_times[3, 1])/2,
         (lookback_times[4, 0]+lookback_times[4, 1])/2,
         (lookback_times[5, 0]+lookback_times[5, 1])/2,
         (lookback_times[6, 0]+lookback_times[6, 1])/2])*1e9
tms = np.array([(sum(sfrs[i, :]*((tbins)*tbins_age))/(10**masses[i])) for i in range(np.shape(sfrs)[0])])

tms_Gyr = tms/1e9
frbs_db["t_m"][0] = compute_quantile(tms_Gyr, [0.5], weights)[0]
frbs_db["t_m_errl"][0] = compute_quantile(tms_Gyr, [0.16], weights)[0] - compute_quantile(tms_Gyr, [0.5], weights)[0]
frbs_db["t_m_erru"][0] = compute_quantile(tms_Gyr, [0.84], weights)[0] - compute_quantile(tms_Gyr, [0.5], weights)[0]

frbs_db.to_csv("host_results.csv")

