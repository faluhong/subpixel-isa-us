"""
    Utility functions for calculating the vegetation index, impervious surface index, and other indices
    (1) NDVI
    (2) kNDVI
    (3) NBR
    (4) EVI
    (5) NDFI
    (6) NDBI
    (7) NDISI
    (8) AF
    (9) ASI
"""

import numpy as np
import time
import os
from os.path import join
import sys
import pandas as pd
from osgeo import gdal, gdalconst, gdal_array
import pysptools
import pysptools.abundance_maps

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, "../.."))
path_basictools = os.path.join(rootpath_project, 'pythoncode')
sys.path.append(path_basictools)


# vegetation index calculation
def ndvi_cal(red_sr, nir_sr):
    """
    calculate the NDVI index

    :param red_sr:
    :param nir_sr:
    :return:
    """
    ndvi = (nir_sr - red_sr) / (nir_sr + red_sr)
    return ndvi


def kndvi_cal(red_sr, nir_sr):
    """
    calculate the kNDVI index, the simplified version of NDVI

    Camps-Valls, Gustau, Manuel Campos-Taberner, Álvaro Moreno-Martínez, Sophia Walther,
    Grégory Duveiller, Alessandro Cescatti, Miguel D. Mahecha et al.
    "A unified vegetation index for quantifying the terrestrial biosphere." Science Advances 7, no. 9 (2021): eabc7447.

    :param red_sr:
    :param nir_sr:
    :return:
    """

    ndvi = (nir_sr - red_sr) / (nir_sr + red_sr)
    kndvi = np.tanh(ndvi)

    return kndvi


def nbr_cal(nir_sr, swir2_sr):
    nbr = (nir_sr - swir2_sr) / (nir_sr + swir2_sr)

    return nbr


def evi_cal(blue_sr, red_sr, nir_sr):
    """
    calculate the EVI index

    :param blue_sr:
    :param red_sr:
    :param nir_sr:
    :return:
    """

    evi = 2.5 * (nir_sr / 10000 - red_sr / 10000) / (nir_sr / 10000 + 6 * red_sr / 10000 - 7.5 * blue_sr / 10000 + 1)

    return evi


def ndfi_cal(blue_sr, green_sr, red_sr, nir_sr, swir1_sr, swir2_sr):
    """
        calculate the NDFI using the pysptools

        The NDFI calculation method, ref to Eric et al. (2020) and Souza et al. (2005)

        Ref:
        [1] https://github.com/GatorSense/HyperspectralAnalysisIntroduction/blob/master/2.%20Unmixing.ipynb
        [2] https://pysptools.sourceforge.io/abundance_maps.html#fully-constrained-least-squares-fcls
        [3] Bullock, E. L., Woodcock, C. E., & Olofsson, P. (2020).
        Monitoring tropical forest degradation using spectral unmixing and Landsat time series analysis. Remote sensing of Environment, 238, 110968.
        [4] Souza Jr, C. M., Roberts, D. A., & Cochrane, M. A. (2005).
        Combining spectral and spatial information to map canopy damage from selective logging and forest fires. Remote Sensing of Environment, 98(2-3), 329-343.

        Args:
            img_unmix: surface reflectance * 10000

    """

    band_stacking = np.array([blue_sr, green_sr, red_sr, nir_sr, swir1_sr, swir2_sr]).T

    endmember_gv = np.array([0.05, 0.09, 0.04, 0.61, 0.3, 0.1])
    endmember_npv = np.array([0.14, 0.17, 0.22, 0.30, 0.55, 0.3])
    endmember_soil = np.array([0.2, 0.3, 0.34, 0.58, 0.6, 0.58])
    endmember_shade = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    endmember_cloud = np.array([0.9, 0.96, 0.8, 0.78, 0.72, 0.65])

    endmembers = np.array([endmember_gv, endmember_npv, endmember_soil, endmember_shade, endmember_cloud])  # 5X6

    fraction = pysptools.abundance_maps.amaps.FCLS(band_stacking, endmembers * 10000)  # type: ignore

    gv = fraction[:, 0]
    npv = fraction[:, 1]
    soil = fraction[:, 2]
    shade = fraction[:, 3]
    cloud = fraction[:, 4]

    gv_shade = gv / (1 - shade)
    ndfi = (gv_shade - npv - soil) / (gv_shade + npv + soil)

    return ndfi


# Imperious surface related index
def ndbi_cal(nir_sr, swir1_sr):
    """
    calculate the NDBI index

    :param nir_sr:
    :param swir1_sr:
    :return:
    """
    ndbi = (swir1_sr - nir_sr) / (swir1_sr + nir_sr)
    return ndbi


def ndisi_cal(green_sr, nir_sr, swir1_sr, thermal_sr):
    """

    calculate the NDISI index (Normalized Difference Impervious Surface Index)

    :param green_sr:
    :param nir_sr:
    :param swir1_sr:
    :param thermal_sr:
    :return:
    """

    mndwi = (green_sr - swir1_sr) / (green_sr + swir1_sr)
    part1 = (mndwi + nir_sr + swir1_sr) / 3
    ndisi = (thermal_sr - part1) / (thermal_sr + part1)
    return ndisi


def af_cal(blue_sr, nir_sr):
    """
    calculate the Artificial surface Factor (AF) index

    :param blue_sr:
    :param nir_sr:
    :return:
    """
    af = (nir_sr - blue_sr) / (nir_sr + blue_sr)
    return af



def asi_cal(blue_sr, green_sr, red_sr, nir_sr, swir1_sr, swir2_sr):
    """
    calculate the ASI index developed by Yongquan Zhao

    Zhao, Yongquan, and Zhe Zhu. "ASI: An artificial surface Index for Landsat 8 imagery."
    International Journal of Applied Earth Observation and Geoinformation 107 (2022): 102703.
    Ref: https://www.sciencedirect.com/science/article/pii/S0303243422000290

    :param blue_sr:
    :param green_sr:
    :param red_sr:
    :param nir_sr:
    :param swir1_sr:
    :param swir2_sr:
    :return:
    """
    af = (nir_sr - blue_sr) / (nir_sr + blue_sr)

    ndvi = (nir_sr - red_sr) / (nir_sr + red_sr)
    msavi_part1 = np.sqrt(np.square(2 * nir_sr + 1) - 8 * (nir_sr - red_sr))
    msavi = (2 * nir_sr + 1 - msavi_part1) * 0.5
    vsf = 1 - ndvi * msavi

    mbi = (swir1_sr - swir2_sr - nir_sr) / (swir1_sr + swir2_sr + nir_sr) + 0.5
    mndwi = (green_sr - swir1_sr) / (green_sr + swir1_sr)
    embi = (mbi - mndwi - 0.5) / (mbi + mndwi + 1.5)
    ssf = 1 - embi

    mf = (blue_sr + green_sr - nir_sr - swir1_sr) / (blue_sr + green_sr + nir_sr + swir1_sr)

    asi = af * vsf * ssf * mf

    return asi


def albedo_cal(blue_sr, red_sr, nir_sr, swir1_sr, swir2_sr):
    """
        shortwave_broadband_albedo = (0.356 * a1 + 0.130 * a3 + 0.373 * a4 + 0.085 * b5 + 0.072 * b7 - 0.072)

        Ref: Liang, Shunlin. "Narrowband to broadband conversions of land surface albedo I: Algorithms."
                                Remote sensing of environment 76, no. 2 (2001): 213-238.

        The TM/ETM+ coefficient values were used. Might cause

    :param blue_sr:
    :param red_sr:
    :param nir_sr:
    :param swir1_sr:
    :param swir2_sr:
    :return:
    """

    shortwave_broadband_albedo = (0.356 * blue_sr
                                  + 0.130 * red_sr
                                  + 0.373 * nir_sr
                                  + 0.085 * swir1_sr
                                  + 0.072 * swir2_sr
                                  - 0.072)

    return shortwave_broadband_albedo





