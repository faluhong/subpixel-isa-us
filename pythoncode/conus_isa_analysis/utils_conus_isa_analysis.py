"""
    utility functions for CONUS ISA analysis
"""

import numpy as np
import os
from os.path import join, exists
import sys
from osgeo import gdal_array

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.util_function.tile_name_convert import (convert_8_tile_names_to_6_tile_names,
                                                        convert_6_tile_names_to_8_tile_names,
                                                        find_h_v_index_from_tile_name)


def load_nlcd_isp(tile_id, year, nlcd_folder='NLCD', rootpath_nlcd_directory=None, flag_filter_ocean=True):
    """
    load the NLCD ISP for a specific tile and year
    This function can be used to load NLCD2021 and Annual NLCD ISP

    :param rootpath_nlcd:
    :param tile_id: the tile id, it should be six characters, such as h03v11, if not six characters, it will be converted to six characters
    :param year:
    :param nlcd_folder: the folder name for NLCD ISP, it could be 'NLCD' or 'NLCD_annual'
    :return:
    """

    if rootpath_nlcd_directory is None:
        rootpath_nlcd = join(rootpath, 'data', nlcd_folder)
    else:
        rootpath_nlcd = join(rootpath_nlcd_directory, 'data', nlcd_folder)

    if len(tile_id) == 8:
        # convert 8 characters to 6 characters
        tile_id_read = convert_8_tile_names_to_6_tile_names(tile_id)
    else:
        tile_id_read = tile_id

    filename_nlcd = join(rootpath_nlcd, tile_id_read, f'nlcd_{tile_id_read}_{year}.tif')
    # print(filename_nlcd)
    if not exists(filename_nlcd):
        print(f'File {filename_nlcd} does not exist \n')
        return None
    else:
        img_nlcd = gdal_array.LoadFile(filename_nlcd)

        if flag_filter_ocean:
            if nlcd_folder == 'NLCD':
                img_nlcd[img_nlcd == 127] = 0  # 127 means ocean in NLCD2021 ISP
            else:
                img_nlcd = gdal_array.LoadFile(filename_nlcd)
                img_nlcd[img_nlcd == 250] = 0  # 250 means ocean in AnnualNLCD ISP

        return img_nlcd


def load_annual_nlcd_isp_stack(list_year, tile_name, nlcd_folder='annual_nlcd',
                               rootpath_nlcd_directory=None,
                               nlcd_filter_ocean_flag=True):
    """
    load the annual NLCD ISP stack

    :param list_year:
    :param tile_name:
    :param nlcd_folder:
    :param rootpath_nlcd_directory:
    :return:
    """

    if rootpath_nlcd_directory is None:
        rootpath_nlcd_directory = rootpath

    nrow, ncol = 5000, 5000
    img_stack_isp_ts = np.zeros((len(list_year), nrow, ncol), dtype=np.uint8)

    for i_year in range(0, len(list_year)):
        year = list_year[i_year]

        img_annual_nlcd_single_year = load_nlcd_isp(tile_id=tile_name,
                                                    year=year,
                                                    nlcd_folder=nlcd_folder,
                                                    rootpath_nlcd_directory=rootpath_nlcd_directory,
                                                    flag_filter_ocean=nlcd_filter_ocean_flag)

        img_stack_isp_ts[i_year, :, :] = img_annual_nlcd_single_year

    return img_stack_isp_ts


def load_conus_isp_stack(list_year, tile_name,
                         filename_prefix='unet_regressor_round_masked',
                         isp_folder='individual_year_tile',
                         rootpath_conus_isp=None):
    """
    load the CONUS ISP time series

    :param list_year:
    :param tile_name: the 8-tile name
    :param filename_prefix: the prefix of the ISP filename, default is 'unet_regressor_round_masked',
                            other options include: 'unet_regressor_round', 'unet_classifier', unet_regressor_round_masked

    :param rootpath_conus_isp: the rootpath of the CONUS ISP

    :return: img_stack_isp_ts: the ISP time series image stack
    """

    if rootpath_conus_isp is None:
        rootpath_conus_isp = rootpath

    nrow, ncol = 5000, 5000
    img_stack_isp_ts = np.zeros((len(list_year), nrow, ncol), dtype=np.uint8)

    for i_year in range(0, len(list_year)):
        year = list_year[i_year]

        if len(tile_name) == 6:
            tile_name_8 = convert_6_tile_names_to_8_tile_names(tile_name)
        else:
            tile_name_8 = tile_name

        filename_isp_tif = join(rootpath_conus_isp, 'results', 'conus_isp', isp_folder,
                                f'{year}', tile_name_8,
                                f'{filename_prefix}_{tile_name_8}_{year}_isp.tif')
        img_isp = gdal_array.LoadFile(filename_isp_tif)

        assert exists(filename_isp_tif), f'{filename_isp_tif} does not exist'

        img_stack_isp_ts[i_year, :, :] = img_isp

    return img_stack_isp_ts


def extract_whole_region_mask(df_conus_state_basic_info, i_state, nrow=5000, ncol=5000, modify_target='state'):
    """
    extract the whole state/MSA mask from the tile

    :param df_conus_state_basic_info:
    :param i_state: the index of the state
    :return:
    """

    region_id = df_conus_state_basic_info['id'].values[i_state]
    region_name = df_conus_state_basic_info['NAME'].values[i_state]

    h_min = df_conus_state_basic_info['h_min'].values[i_state]
    h_max = df_conus_state_basic_info['h_max'].values[i_state]
    v_min = df_conus_state_basic_info['v_min'].values[i_state]
    v_max = df_conus_state_basic_info['v_max'].values[i_state]

    list_tile_name = df_conus_state_basic_info['tile_name'].values[i_state].split(';')
    assert len(list_tile_name) > 0, f'{region_name} has no tile'

    img_state_id_whole = np.zeros(((v_max - v_min + 1) * nrow, (h_max - h_min + 1) * ncol), dtype=np.int16)

    for i_tile, tile_name in enumerate(list_tile_name):

        if (modify_target == 'state') | (modify_target == 'county'):
            filename_state_id_tile = join(rootpath, 'data', 'shapefile', 'CONUS_boundary', f'tl_2023_us_{modify_target}',
                                          f'individual_tile_{modify_target}',
                                          f'tl_2023_us_{modify_target}_{tile_name}.tif')

        elif modify_target == 'msa':
            filename_state_id_tile = join(rootpath, 'data', 'urban_pulse', 'shapefile', 'cb_2015_us_cbsa_500k',
                                          f'individual_tile_{modify_target}',
                                          f'cb_2015_us_cbsa_500k_ard_{tile_name}.tif')
        else:
            raise ValueError(f'Unknown modify_target: {modify_target}')


        img_state_id_tile = gdal_array.LoadFile(filename_state_id_tile)

        h_index, v_index = find_h_v_index_from_tile_name(tile_name)

        img_state_id_whole[(v_index - v_min) * nrow: (v_index - v_min + 1) * nrow, (h_index - h_min) * ncol: (h_index - h_min + 1) * ncol] = img_state_id_tile

    mask_whole_state = img_state_id_whole == region_id

    return mask_whole_state


def get_isp_change_stats(img_stack_ts_is_pct):
    """
    get the ISP change statistics, i.e., the pixel count of ISP change statistics, 2D array, ISP change from year 1 to year 2

    :param img_stack_ts_is_pct:
    :return:
    """

    assert len(np.shape(img_stack_ts_is_pct)) == 3, 'The shape of the input ISP image stack is not correct'
    assert np.shape(img_stack_ts_is_pct)[0] == 2, 'The number of input ISP images is not 2'

    img_year_1 = img_stack_ts_is_pct[0, :, :]
    img_year_2 = img_stack_ts_is_pct[1, :, :]

    # This for loop is very slow, it takes ~400 seconds to finish running use np.histogram2d instead
    # img_isp_change_stats = np.zeros((101, 101), dtype=np.int32)
    #
    # for i_isp_year_1 in range(0, 101):
    #     for i_isp_year_2 in range(0, 101):
    #         mask_year_1 = (img_year_1 == i_isp_year_1) & (img_year_2 == i_isp_year_2)
    #         img_isp_change_stats[i_isp_year_1, i_isp_year_2] = np.sum(mask_year_1)
    #         del mask_year_1

            # print(i_isp_year_1, i_isp_year_2, img_isp_change_stats[i_isp_year_1, i_isp_year_2])

    # np.histogram2d is much faster than the for loop, it takes around 2 seconds to finish running
    img_isp_change_stats = np.histogram2d(img_year_1.ravel(), img_year_2.ravel(), bins=101, range=[[0, 100], [0, 100]])[0].astype(np.int32)

    return img_isp_change_stats







