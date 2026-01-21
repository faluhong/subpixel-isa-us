"""
    utility functions for computing centroid trajectory statistics

    (1) calculate the annual centroid movement distance
    (2) calculate the linearity ratio of the trajectory
    (3) calculate the annual centroid movement angle change

    (4) plot the annual centroid movement distance trend for all targets
    (5) plot the annual centroid movement distance trend for a single target
"""

import numpy as np
import os
from os.path import join, exists
import sys
from osgeo import gdal, gdal_array, gdalconst
import geopandas as gpd
from geopy.distance import geodesic

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def calculate_annual_centroid_distance_move(array_latitude, array_longitude):
    """
        calculate the annual centroid movement distance for each target (state, MSA)

        Parameters
        ----------
        array_latitude: 1D numpy array
            array of latitude of centroid for each year
        array_longitude: 1D numpy array
            array of longitude of centroid for each year

        Returns
        -------
        array_annual_distance_m: 1D numpy array (size -1)
            array of annual centroid movement distance (in meters)
    """

    list_annual_distance_m = []
    for i_year in range(len(array_latitude) - 1):
        dist_m = geodesic((array_latitude[i_year], array_longitude[i_year]),
                          (array_latitude[i_year + 1], array_longitude[i_year + 1])).meters
        list_annual_distance_m.append(dist_m)

    array_annual_distance_m = np.array(list_annual_distance_m)

    return (array_annual_distance_m)


def calculate_linearity_ratio(array_latitude, array_longitude):
    """
        Calculate linearity ratio of the trajectory

        :param array_latitude:
        :param array_longitude:
        :return:
    """

    # Compute direct start-to-end distance
    direct_dist = geodesic((array_latitude[0], array_longitude[0]), (array_latitude[-1], array_longitude[-1])).meters

    # Compute total distance along the trajectory
    array_annual_distance_m = calculate_annual_centroid_distance_move(array_latitude, array_longitude)
    total_dist = np.sum(array_annual_distance_m)

    # Linearity ratio
    linearity = direct_dist / total_dist

    return linearity











