# standard regular-use python packages
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# standard imports for my work
from read_transients_data import *
from correct_redshift_evolution import *
from generate_bkg_galaxies import *
from helper_functions import *
from ndtest import ks2d2s


def return_2d_kstest_results(arr1, arr2, arr3, arr4):
    """
    Perform a 2D Kolmogorov-Smirnov test between two datasets and multiple sets 
    of coordinates.

    Parameters:
    ----------
    arr1 : array-like
        First dataset.
    arr2 : array-like
        Second dataset.
    arr3 : array-like
        Array of x-coordinates for the sets to test against (e.g., theoretical 
        distribution).
    arr4 : array-like
        Array of y-coordinates for the sets to test against.

    Returns:
    -------
    float
        50th percentile of P-values resulting from the 2D KS test.
    """
    Ps = []
    for i in tqdm(range(len(arr3))):
        P, D = ks2d2s(arr1, arr2, arr3[i], arr4[i], extra=True)
        Ps.append(P)
    return np.percentile(Ps, 50)



def classify_bpt(log_OIII_Hbeta, log_NII_Halpha, log_SII_Halpha, log_OI_Halpha):
    """
    Classifies galaxies based on the BPT diagram using emission line ratios.

    Parameters:
    ----------
    log_OIII_Hbeta : float
        Logarithm of [O III] / H-beta emission line ratio.
    log_NII_Halpha : float
        Logarithm of [N II] / H-alpha emission line ratio.
    log_SII_Halpha : float
        Logarithm of [S II] / H-alpha emission line ratio.
    log_OI_Halpha : float
        Logarithm of [O I] / H-alpha emission line ratio.

    Returns:
    -------
    tuple
        Three strings indicating the classification of the galaxy based on 
        different BPT diagrams:
        - galaxy_class1: Classification based on the Ke01 diagram.
        - galaxy_class2: Classification based on the Ka03 diagram (2nd method).
        - galaxy_class3: Classification based on the Ka03 diagram (3rd method).
          Possible values for each classification: "HII", "LINER", "Seyfert".

    Notes:
    ------
    - Uses specific threshold equations derived from literature (Ke01, Ka03 
      methods) to classify galaxies.
    - Assumes input parameters are logarithms of appropriate emission line 
      ratios.
    """

    log_OIII_Hbeta_Ke01 = (0.61 / (log_NII_Halpha - 0.05)) + 1.3
    metric_2 = -(0.05932203389830515 * 0.7464452280741671) + (1.2711864406779658 * -0.4467637356387215) + log_OIII_Hbeta * ((0.05932203389830515 - 1.2711864406779658) / (-0.4467637356387215 - 0.7464452280741671))

    if log_OIII_Hbeta < log_OIII_Hbeta_Ke01 and log_NII_Halpha < 0:
        galaxy_class1 = "HII"
    elif log_NII_Halpha < metric_2:
        galaxy_class1 = "LINER"
    else:
        galaxy_class1 = "Seyfert"
    
    log_OIII_Hbeta_Ka03_2 = (0.72 / (log_SII_Halpha - 0.32)) + 1.3
    log_OIII_Hbeta_Ke03_1 = (1.89 * log_SII_Halpha) + 0.76

    if log_OIII_Hbeta < log_OIII_Hbeta_Ka03_2:
        galaxy_class2 = "HII"
    elif log_OIII_Hbeta < log_OIII_Hbeta_Ke03_1:
        galaxy_class2 = "LINER"
    else:
        galaxy_class2 = "Seyfert"

    log_OIII_Hbeta_Ka03_3 = (0.73 / (log_OI_Halpha + 0.59)) + 1.33
    log_OIII_Hbeta_Ke03_2 = (1.18 * log_OI_Halpha) + 1.30

    if log_OIII_Hbeta < log_OIII_Hbeta_Ka03_3 and log_OI_Halpha < -1:
        galaxy_class3 = "HII"
    elif log_OIII_Hbeta < log_OIII_Hbeta_Ke03_2:
        galaxy_class3 = "LINER"
    else:
        galaxy_class3 = "Seyfert"

    return galaxy_class1, galaxy_class2, galaxy_class3


