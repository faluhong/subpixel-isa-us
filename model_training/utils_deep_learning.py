"""
    functions for deep learning ISP project
"""

import numpy as np
import os
from os.path import join
import sys
import pandas as pd
from osgeo import gdal, gdalconst, gdal_array
import scipy.io as scio
import json as js
import click
import fiona
import logging
import joblib
import geopandas as gpd
import re

pwd = os.getcwd()
rootpath_project = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)


def read_cold_variable(predictor_variable_folder, tile_name, year, central_reflectance_flag='change', rootpath_project_folder=None):
    """
    Get the training data

    The training variables include: (1) overall reflectance on July-1st for each year; (2) a1, b2, c1, RMSE for each year
    Total 35 features (5 variables * 7 bands)

    :param predictor_variable_folder:
    :param tile_name:
    :param year:
    :param central_reflectance_flag:  # flag to use the segment stable or annual change central reflectance, 'stable' or 'change', default as 'change'

    :return:
    """

    if rootpath_project_folder is None:
        path_sr = join(rootpath_project, 'data', predictor_variable_folder, tile_name, str(year))
    else:
        path_sr = join(rootpath_project_folder, 'data', predictor_variable_folder, tile_name, str(year))

    img_a1 = gdal_array.LoadFile(join(path_sr, f'{tile_name}_a1_{year}.tif'))
    img_b1 = gdal_array.LoadFile(join(path_sr, f'{tile_name}_b1_{year}.tif'))

    img_c1 = gdal_array.LoadFile(join(path_sr, f'{tile_name}_c1_{year}.tif'))
    img_rmse = gdal_array.LoadFile(join(path_sr, f'{tile_name}_RMSE_{year}.tif'))

    if central_reflectance_flag == 'change':
        img_sr_change = gdal_array.LoadFile(join(path_sr, f'{tile_name}_SR_change_{year}.tif'))
        img_training = np.concatenate([img_sr_change, img_a1, img_b1, img_c1, img_rmse])
    else:
        img_sr_stable = gdal_array.LoadFile(join(path_sr, f'{tile_name}_SR_stable_{year}.tif'))
        img_training = np.concatenate([img_sr_stable, img_a1, img_b1, img_c1, img_rmse])

    return img_training


def read_global_normalization_boundary(norm_boundary_folder='maximum_minimum_ref'):
    """
        read the global normalization boundary

        :param norm_boundary_folder: the folder name of the reference files, default as 'maximum_minimum_ref'
                                maximum_minimum_ref: previous one based on the selected tiles
                                maximum_minimum_ref_conus: updated one based on the CONUS tiles
    """

    path_output = join(rootpath_project, 'results', 'deep_learning', norm_boundary_folder)
    filename_max = join(path_output, 'sum_max_boundary.npy')
    filename_min = join(path_output, 'sum_min_boundary.npy')

    array_max_ref = np.load(filename_max)
    array_min_ref = np.load(filename_min)

    return array_max_ref, array_min_ref


def predictor_normalize(img_x_training, norm_boundary_folder='maximum_minimum_ref'):
    """
    normalize the predictor variables

    :param img_x_training:
    :param norm_boundary_folder: the folder name storing the normalization boundary, default as 'maximum_minimum_ref'
    :return:
    """

    array_maximum, array_minimum = read_global_normalization_boundary(norm_boundary_folder=norm_boundary_folder)

    img_return = np.zeros(np.shape(img_x_training), dtype=float)

    for i_feature in range(0, np.shape(img_x_training)[0]):

        max_boundary = array_maximum[i_feature]
        min_boundary = array_minimum[i_feature]

        img_per_feature = img_x_training[i_feature, :, :].copy()

        img_per_feature[np.isnan(img_per_feature)] = 0
        img_per_feature[img_per_feature >= max_boundary] = max_boundary
        img_per_feature[img_per_feature <= min_boundary] = min_boundary

        if np.max(img_per_feature) == np.min(img_per_feature):
            # if the maximum and minimum are the same, then the image is a constant image
            img_per_feature = 0
        else:
            img_per_feature = (img_per_feature - min_boundary) / (max_boundary - min_boundary)

        img_return[i_feature, :, :] = img_per_feature

    return img_return


def read_topography_data(tile_name, rootpath_project_folder=None):
    """
    read the topography data (DEM, slope, aspect) for the tile

    :param tile_name:
    :param rootpath_project_folder:
    :return:
    """


    if rootpath_project_folder is None:
        path_topography = join(rootpath_project, 'data', 'topography')
    else:
        path_topography = join(rootpath_project_folder, 'data', 'topography')

    filename_dem = join(path_topography, tile_name, f'{tile_name}_dem.tif')
    filename_slope = join(path_topography, tile_name, f'{tile_name}_slope.tif')
    filename_aspect = join(path_topography, tile_name, f'{tile_name}_aspect.tif')

    img_dem = gdal_array.LoadFile(filename_dem)
    img_slope = gdal_array.LoadFile(filename_slope)
    img_aspect = gdal_array.LoadFile(filename_aspect)

    return img_dem, img_slope, img_aspect


def read_global_topography_boundary(norm_boundary_folder='maximum_minimum_ref'):
    """
        read the CONUS topography boundary for normalization
        (1) elevation
        (2) slope
        (3) aspect
    """

    filename_topography_boundary_stats = join(rootpath_project, 'results', 'deep_learning', norm_boundary_folder, 'conus_topography_boundary.csv')
    df_topography_boundary = pd.read_csv(filename_topography_boundary_stats)

    # the upper and lower boundary of the topography information is defined based on the global boundary, not the percentile
    array_upper_elevation = df_topography_boundary['elevation_max'].values[-1]
    array_lower_elevation = df_topography_boundary['elevation_min'].values[-1]

    array_upper_slope = df_topography_boundary['slope_max'].values[-1]
    array_lower_slope = df_topography_boundary['slope_min'].values[-1]

    array_upper_aspect = df_topography_boundary['aspect_max'].values[-1]
    array_lower_aspect = df_topography_boundary['aspect_min'].values[-1]

    # array_upper_elevation = df_topography_boundary['elevation_99'].values[-1]
    # array_lower_elevation = df_topography_boundary['elevation_1'].values[-1]
    #
    # array_upper_slope = df_topography_boundary['slope_99'].values[-1]
    # array_lower_slope = df_topography_boundary['slope_1'].values[-1]
    #
    # array_upper_aspect = df_topography_boundary['aspect_99'].values[-1]
    # array_lower_aspect = df_topography_boundary['aspect_1'].values[-1]

    return array_upper_elevation, array_lower_elevation, array_upper_slope, array_lower_slope, array_upper_aspect, array_lower_aspect


def topography_normalize(img_dem, img_slope, img_aspect, norm_boundary_folder='maximum_minimum_ref'):
    """
    normalize the topography data

    :param img_dem:
    :param img_slope:
    :param img_aspect:
    :param norm_boundary_folder: the folder name storing the normalization boundary, default as 'maximum_minimum_ref'
    :return:
    """

    # read the global topography boundary
    (array_upper_elevation, array_lower_elevation,
     array_upper_slope, array_lower_slope,
     array_upper_aspect, array_lower_aspect) = read_global_topography_boundary(norm_boundary_folder=norm_boundary_folder)

    # clip the topography data
    img_dem[img_dem < array_lower_elevation] = array_lower_elevation
    img_dem[img_dem > array_upper_elevation] = array_upper_elevation

    img_slope[img_slope < array_lower_slope] = array_lower_slope
    img_slope[img_slope > array_upper_slope] = array_upper_slope

    img_aspect[img_aspect < array_lower_aspect] = array_lower_aspect
    img_aspect[img_aspect > array_upper_aspect] = array_upper_aspect

    # normalization the topography information
    img_dem = (img_dem - array_lower_elevation) / (array_upper_elevation - array_lower_elevation)
    img_slope = (img_slope - array_lower_slope) / (array_upper_slope - array_lower_slope)
    img_aspect = (img_aspect - array_lower_aspect) / (array_upper_aspect - array_lower_aspect)

    # add one dimension to the topography data
    img_dem = np.expand_dims(img_dem, axis=0)
    img_slope = np.expand_dims(img_slope, axis=0)
    img_aspect = np.expand_dims(img_aspect, axis=0)

    return img_dem, img_slope, img_aspect


def get_proj_info(tile_name):
    """
    get the projection information from Landsat ARD tiles

    :param tile_name:
    :return:
    """

    # Regular expression to find all integers
    matches = re.findall(r'\d+', tile_name)

    # Convert the extracted strings to integers
    integers = [int(match) for match in matches]

    h_index = integers[0]
    v_index = integers[1]

    filename_conus_ard_grid = join(rootpath_project, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)

    gpd_ard_tile = gpd_ard[(gpd_ard['h'] == h_index) & (gpd_ard['v'] == v_index)]

    proj_ard = gpd_ard.crs.to_wkt()

    geo_transform = (gpd_ard_tile.total_bounds[0], 30, 0, gpd_ard_tile.total_bounds[3], 0, -30)

    return proj_ard, geo_transform


def add_pyramids_color_in_nlcd_isp_tif(filename_tif, list_overview=None):
    """
        add pyramids and color table in the tif file based on the NLCD ISP color table
    """

    if list_overview is None:
        list_overview = [2, 4, 8, 16, 32, 64]

    dataset = gdal.Open(filename_tif, gdal.GA_Update)

    # Generate overviews/pyramids
    # The list [2, 4, 8, 16, 32] defines the downsampling factors for the overviews
    dataset.BuildOverviews(overviewlist=list_overview)

    # Get the first band of the image
    band = dataset.GetRasterBand(1)

    # Create a new color table
    color_table = gdal.ColorTable()

    # Set the color for each value in the color table
    filename_nlcd_color = join(rootpath_project, 'figure', 'NLCD_color_table.xlsx')
    df_nlcd_color = pd.read_excel(filename_nlcd_color)

    for i in range(0, len(df_nlcd_color)):
        color_string = df_nlcd_color['color'].values[i]

        color_string = color_string.lstrip('#')

        color_tuple = tuple(int(color_string[i:i + 2], 16) for i in (0, 2, 4))
        color_tuple = color_tuple + (255,)

        color_table.SetColorEntry(i, color_tuple)

    # Assign the color table to the band
    band.SetRasterColorTable(color_table)

    # Save the changes and close the dataset
    dataset = None

    return None


if __name__ == '__main__':
    year = 1985
    tile_name = 'h027v008'

    # img_blue, img_green, img_red, img_nir, img_swir1, img_swir2, img_thermal, img_rmse, img_harmonic_coef \
    #     = load_surface_reflectance(tile_name, year)
    #
    # img_nlcd = load_nlcd(tile_name)


