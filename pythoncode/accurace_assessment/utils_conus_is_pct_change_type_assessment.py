"""
    utility function to assess the accuracy of IS percentage value and IS change in the CONUS
"""

import os
from os.path import join
import sys
import pandas as pd
import numpy as np
import fiona
import geopandas as gpd
from pyproj import CRS, Transformer
import simplekml

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def get_weight_for_whole_conus_is_change_type(array_target_year, data_flag, isp_folder, rootpath_project_folder=None):
    """
        get the weight of each IS change type for the whole CONUS through reading the pre-calculated IS change type count

        Args:
            array_target_year (np.array): the target year with the mapped ISP
            data_flag (str): the data flag for the analysis
            isp_folder (str): the folder name for the ISP data
            rootpath_project_folder: the root path of the project folder
    """

    if rootpath_project_folder is None:
        rootpath_project_folder = rootpath

    output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag, isp_folder)
    df_is_change_sum_conus = pd.read_csv(join(output_folder, 'conus_is_change_type_count_with_sm.csv'))

    df_is_change_sum_conus_cal = df_is_change_sum_conus[np.isin(df_is_change_sum_conus['year_1'].values, (array_target_year[0:-1]))].copy()

    assert len(df_is_change_sum_conus_cal) > 0, 'No data for the selected year range'

    if len(df_is_change_sum_conus_cal) > 1:
        array_count = df_is_change_sum_conus_cal.iloc[:, 2:9].values.sum(axis=0)
    else:
        array_count = np.ravel(df_is_change_sum_conus_cal.iloc[:, 2:9].values)

    array_count = array_count.astype(float)

    array_weight = array_count / np.nansum(array_count)

    return (array_weight, array_count)


def get_tile_row_col_id_from_conus_location(row_id_conus, col_id_conus):
    """
    get the tile name, row id, and column id in the tile based on the CONUS row and column id

    :param row_id_conus:
    :param col_id_conus:
    :return:
    """
    v_index = (row_id_conus) // 5000
    h_index = (col_id_conus) // 5000

    tile_name = f'h{h_index:03d}v{v_index:03d}'  # 8 characters

    row_id_tile = row_id_conus - v_index * 5000
    col_id_tile = col_id_conus - h_index * 5000

    return (tile_name, row_id_tile, col_id_tile)


def get_lat_long_coords(tile_name, row_id, col_id):
    """
    get the latitude and longitude coordinates of the pixel based on the tile, row, and column id

    :param tile_name:
    :param row_id:
    :param col_id:
    :return:
    """

    from pythoncode.model_training.utils_deep_learning import get_proj_info
    proj_ard, geo_transform = get_proj_info(tile_name=tile_name)

    x_left_up = geo_transform[0] + col_id * geo_transform[1]
    y_left_up = geo_transform[3] + row_id * geo_transform[5]

    x_left_down = x_left_up
    y_left_down = geo_transform[3] + (row_id + 1) * geo_transform[5]

    x_right_up = geo_transform[0] + (col_id + 1) * geo_transform[1]
    y_right_up = y_left_up

    x_right_down = geo_transform[0] + (col_id + 1) * geo_transform[1]
    y_right_down = geo_transform[3] + (row_id + 1) * geo_transform[5]

    proj_wgs84 = CRS("WGS84")
    transformer = Transformer.from_proj(proj_ard, proj_wgs84)

    latitude_left_up, longitude_left_up = transformer.transform(x_left_up, y_left_up)
    latitude_left_down, longitude_left_down = transformer.transform(x_left_down, y_left_down)
    latitude_right_up, longitude_right_up = transformer.transform(x_right_up, y_right_up)
    latitude_right_down, longitude_right_down = transformer.transform(x_right_down, y_right_down)

    coords = [(longitude_left_up, latitude_left_up),
              (longitude_left_down, latitude_left_down),
              (longitude_right_down, latitude_right_down),
              (longitude_right_up, latitude_right_up),
              (longitude_left_up, latitude_left_up)
              ]

    return coords


def prepare_evaluation_data(df_interpretation, dict_is_change_type, map_column_name='stratum'):
    """
        prepare the data for evaluation, i.e., get the map and reference data for the confusion matrix

        :param df_interpretation: the interpretation spreadsheet
        :return:
    """

    reverse_dict_is_change_type = {v: int(k) for k, v in dict_is_change_type.items()}

    # get the map results
    array_map = df_interpretation[map_column_name].values
    array_map = np.array([reverse_dict_is_change_type.get(i, -999) for i in array_map])

    # get the reference results
    array_reference = df_interpretation['interpretation_is_change_type'].values
    array_reference = np.array([reverse_dict_is_change_type.get(i, -999) for i in array_reference])

    # final data to report the accuracy
    mask_exclude = (array_map == -999) | (array_reference == -999)

    array_map_final = array_map[~mask_exclude]
    array_reference_final = array_reference[~mask_exclude]

    # count-based confusion matrix
    categories = np.arange(1, len(dict_is_change_type) + 1)

    # define the categories to avoid missing categories in the confusion matrix
    array_map_final = pd.Categorical(array_map_final, categories=categories)
    array_reference_final = pd.Categorical(array_reference_final, categories=categories)

    return (array_map_final, array_reference_final)





