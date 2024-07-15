<img width="1504" alt="dsa110" src="https://github.com/user-attachments/assets/edddc96c-2a61-4db0-8993-294edf026b7c">

# DSA FRBs Host Galaxies Analysis

## Installation Requirements

The specific requirements for running these scripts can be installed by `pip install -r requirements.txt`. These include:

- [scipy](https://scipy.org/install/)
- [astropy](https://astropy.readthedocs.org/en/stable/)
- [torch](https://pypi.org/project/torch/)
- [astro-prospector](https://pypi.org/project/astro-prospector/)
- [astro-sedpy](https://github.com/bd-j/sedpy)
- [dynesty](https://dynesty.readthedocs.io/en/latest/)
- [sedpy](https://github.com/bd-j/sedpy)
- [HDF5](https://www.hdfgroup.org/HDF5/) and [h5py](http://www.h5py.org)
- [FSPS](https://github.com/cconroy20/fsps) and [python-FSPS](https://github.com/dfm/python-FSPS)
- [PPXF](https://pypi.org/project/ppxf/)




## Description

**Data Directories**

* `frbs_data` Contains all the relevant data for FRBs used in our work, including
  * Literature FRB host galaxies and offset measurements data as published in literature (relevant citations below)
  * P(z|DM) function from [Zhang et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...906...49Z) `p_of_dm_z.npy`
  * DSA data, provided in the following structure **[To be made public after publication]**:
     * FRB host galaxy properties, line emission flux measurements and offset measurements `dsa_frbs.csv`

* `galaxies_data` Contains all the data relevant for background galaxies, including
  * Galaxy stellar population properties of COSMOS galaxy catalog `prospector_cosmos_catalog.dat` ([Laigle et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJS..224...24L/abstract), [Leja et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract), [Leja et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...936..165L/abstract)) and 3D-HST galaxy catalog `prospector_3dhst_catalog.dat` ([Skelton et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014ApJS..214...24S/abstract), [Leja et al. (2019)](https://ui.adsabs.harvard.edu/abs/2020IAUS..352...99L/abstract), [Leja et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract), [Leja et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...936..165L/abstract))
  * Galaxy spatial number density as a function of galaxy apparent magnitudes from [Driver et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...827..108D) `driver_et_al.fits` for computing the probability of chance coincidence
  * SFR and stellar mass weighted background galaxy population empirical distribution as sampled using methods developed in our work `local_univ_bkg.csv` for reference
  * Trained normalizing flow as computed by [Leja et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...936..165L/abstract) `trained_flow_nsamp100.pth` for fitting the galaxy star-forming main sequence and galaxy number density in logM - logSFR - z space
  * Galaxy mass-metallicity relation from [Gallazzi et al. (2005)](https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G) `gallazzi_05_massmet.txt` used in our SED analysis as a prior

* `other_transients_data` Contains the host galaxy properties and offset measurements derived from literature of other transients. The relevant citations are listed below.

**Python Scripts**

* `read_transients_data.py` Methods for reading in host galaxy properties and offset distributions of various transient classes, such as FRBs, long-duration GRBs, SLSNe, CCSNe, Type Ia SNe short-duration GRBs, ULX sources, Milky Way satellites and Milky Way Globular clusters. The relevant literature data sources are listed below in citations section.
  
* `sed_analysis_funcs.py` Methods for performing SED analysis of galaxies using [Prospector](https://github.com/bd-j/prospector), including the mass-metallicity prior, normalizing spectrum using photometry, building observations dictionary, building the galaxy SED model and relevant functions for the same.

* `sed_analysis.py` Performs the SED analysis of a galaxy using the methods defined in `sed_analysis_funcs.py`. For example, `python sed_analysis.py --frb mark` performs SED analysis for FRB 20220319D (Mark) by using the galaxy photometry listed in `frb_obs.csv` and the provided spectrum at `spec.csv`, generates relevant plots for inspection in the FRB directory (`frb_host_sharma2024/frbs_data/dsa_results/mark/`), saves the host galaxy properties in the file `host_results.csv`, and saves the calibrated spectrum in the file `spec_calibrated.csv`.

* `sample_nf_probability_density.py` and `torch_light.py` Methods for reproducing and interpreting the trained normalizing flow describing the galaxy star-forming sequence of [Leja et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...936..165L/abstract), including methods for computing the stellar mass-complete limits. The trained flow is stored in `frb_host_sharma2024/galaxies_data/trained_flow_nsamp100.pth`.

* `generate_bkg_galaxies.py` Methods for generating the background galaxy population using the galaxy stellar mass function fits of [Leja et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract) and the galaxy star-forming main sequence and galaxy number density in logM - logSFR - z space fits as presented in [Leja et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract). 

* `correct_redshift_evolution.py` Methods for correcting stellar population properties (logM and logSFR) of transients for redshift evolution to allow comparisons at z = 0, without being impacted by evolution of galaxies.

* `bpt_utils.py` Methods for classifying galaxies as star-forming, LINERs or Seyferts using [Kewley et al. 2001](https://ui.adsabs.harvard.edu/abs/2001ApJ...556..121K), [Kauffman et al. (2003)](https://ui.adsabs.harvard.edu/abs/2003MNRAS.346.1055K) BPT classification schemes and performing 2D KS tests.

* `helper_functions.py` Methods for plotting, generating quantiles and samples.

**Jupyter Notebooks**

* `association_capability_test.ipynb` Methods for testing the host association capabilites of DSA, given the localization uncertainties and limiting magnitudes of optical surveys used, primarily by using the probability of chance coincidence.

* `rarity_of_wrong_associations.ipynb` Methods for quantifying the rarity of existence of multiple candidate galaxies within the FRB localization regions using the galaxy stellar mass function and accounting for galaxy clustering.

* `path_analysis.ipynb` An example PATH host association exercise for DSA FRBs with deepest available data of Legacy Survey catalogs. This script can be modified to achieve PATH associations using PS1, WIRC or WaSP data by using SExtractor package.

* `plot_host_cutouts.ipynb` An example notebook for plotting DSA FRB host galaxies.

* `plot_ppxf_fits.ipynb` An example notebook for measuring host galaxy redshift and emission line fluxes using [PPXF](https://pypi.org/project/ppxf/).
  
* `plot_sed_fits.ipynb` An example notebook for plotting SED fits of DSA FRB host galaxies.
  
* `bpt_analysis.ipynb` Methods for plotting the BPT diagram and performing relevant KS tests.
  
* `compare_with_background_galaxy_population.ipynb` Generates background galaxy population using the methods defined in `generate_bkg_galaxies.py`, generates SFR and stellar mass weighted background galaxy distributions and performs KS tests to compare them with FRBs in various redshift bins.

* `frbs_are_biased_SF_tracer.ipynb` Compares FRBs with CCSNe in local redshift bin, fits SFR and metallicity weighted background galaxy population to FRBs to show that FRBs are a biased tracer of star-formation, preferentially favoring metal-rich environments.

* `significance_of_low_mass_hosts_deficit.ipynb` Quantifies the rarity of all CCSNe occuring in massive, FRB-like galaxies.

* `compare_with_cosmos_3dhst.ipynb` Compare FRBs with galaxy star-forming main sequence constructed using COSMOS and 3D-HST galaxy catalogs.

* `galaxy_properties_distributions.ipynb` Compares FRB host galaxy properties with COSMOS and 3D-HST galaxy catalogs.

* `compare_with_other_transients (local universe).ipynb` Compares FRB host galaxy properties and offset measurments with other transients in the local universe.

* `offset_distributions.ipynb` Compares FRB offset measurments with other transients.

* `compare_with_other_transients.ipynb` Compares FRB host galaxy properties with other transients.




## Relevant Citations

If using the **background galaxy population generation procedures** developed in our work, please cite Sharma et al. (2024), [Leja et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract), [Leja et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...936..165L/abstract) and [https://github.com/jrleja/sfs_leja_trained_flow](https://github.com/jrleja/sfs_leja_trained_flow).

If using the **DSA-110 FRB host galaxies catalog** published in our work or **publicly available transients catalogs** used in our work, please cite the relevant papers from the following list:

- COSMOS galaxy properties catalog: [Laigle et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJS..224...24L/abstract), [Leja et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract), [Leja et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...936..165L/abstract)
- 3D-HST galaxy properties catalog: [Skelton et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014ApJS..214...24S/abstract), [Leja et al. (2019)](https://ui.adsabs.harvard.edu/abs/2020IAUS..352...99L/abstract), [Leja et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract), [Leja et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...936..165L/abstract)
- DSA-110 discovered FRB host galaxy properties and offset measurements: Sharma et al. (2024), [Law & Sharma et al. (2023)](https://ui.adsabs.harvard.edu/abs/2024ApJ...967...29L/abstract)
- CHIME discovered FRB host galaxy properties: [Bhardwaj et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv231010018B/abstract)
- ASKAP/CRAFT, MeerKAT, CHIME, Arecibo and Parkes discovered FRB host galaxy properties: [Gordon et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJ...954...80G/abstract)
- FRB offset measurements: [Mannings et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...917...75M/abstract), [Woodland et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv231201578W/abstract)
- Type Ia supernovae host galaxy properties: [Lampeitl et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...722..566L/abstract), [Childress et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013ApJ...770..108C/abstract)
- Type Ia supernovae offset measurements: [Uddin et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...901..143U/abstract)
- ULX Sources host properties and offset measurements: [Kovlakas et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.498.4790K/abstract)
- CCSNe host galaxy properties: [Schulze et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJS..255...29S/abstract)
- CCSNe offset measurements: [Schulze et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJS..255...29S/abstract), [Kelly & Kirshner (2012)](https://ui.adsabs.harvard.edu/abs/2012ApJ...759..107K/abstract)
- SLSNe host galaxy properties: [Schulze et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJS..255...29S/abstract), [Taggart & Perley (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.3931T/abstract)
- SLSNe offset measurements: [Lunnan et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...804...90L/abstract)
- Short-duration GRBs host galaxy properties: [Nugent et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...940...57N/abstract)
- Short-duration GRBs offset measurements: [Fong et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...940...56F/abstract)
- Long-duration GRB host galaxy properties: [Vergani et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015A&A...581A.102V), [Taggart & Perley (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.3931T/abstract)
- Satellites of Milky Way offset measurements: [Drlica-Wagner et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...893...47D/abstract)
- Globular clusters of Milky Way offset measurements: [Harris (1996)](https://ui.adsabs.harvard.edu/abs/1996AJ....112.1487H/abstract)
- Long-duration GRB offset measurements: [Blanchard et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...817..144B/abstract)
