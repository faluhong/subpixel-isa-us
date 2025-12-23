"""
    calculate the impervious surface area for each state from 1985 to 2022
    
    Recommend to use 96 GBs to run this code to avoid memory issue in large states, such as Texas
    The maximum memory usage is about 65 GBs
    
    Using 20 cores for parallel computing, the running time 50 minutes for all states from 1985 to 2022/2023
"""

import numpy as np
import os
from os.path import join, exists
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import matplotlib
import seaborn as sns
import geopandas as gpd
import logging
import click

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from evaluation.utils_evaluation import convert_8_tile_names_to_6_tile_names, convert_6_tile_names_to_8_tile_names

from conus_isp_production.utils_get_conus_tile_name import get_conus_tile_name
from conus_isp_production.clip_conus_isp import find_h_v_index_from_tile_name

from auxiliary_data_process.prepare_state_county_basic_info import (extract_whole_region_mask)

from post_processing.isp_ts_post_processing import load_conus_isp_stack
from conus_isp_analysis.tile_is_change_stats_output import load_annual_nlcd_isp_stack

from evaluation.utils_plot_isp import plot_isp_single
from conus_isp_analysis.tile_is_change_stats_output import get_isp_change_stats

# from analysis.utils_isp_change_stats_analysis import (plot_is_pct_change_between_two_years, plot_isp_change_type_ts,
#                                                       get_isp_change_stats_summary_single_tile_all_year,
#                                                       generate_isp_change_summary_dataframe,
#                                                       read_isp_change_stats_output_file_single_tile_year)


def load_state_isp_stack(df_conus_state_basic_info, i_state, array_year,
                         data_flag, isp_folder, filename_prefix, nlcd_folder, nlcd_filter_ocean_flag,
                         rootpath_conus_isp=None,
                         rootpath_nlcd_directory=None,
                         ):
    """
    load the ISP stack for one state

    :param df_conus_state_basic_info:
    :param i_state:
    :param array_year: such as np.arange(1985, 1987)
    :param data_flag: conus_isp or annual_nlcd
    :param isp_folder:
    :param filename_prefix:
    :param nlcd_folder:
    :param nlcd_filter_ocean_flag:
    :param rootpath_conus_isp:
    :param rootpath_nlcd_directory:

    :return:
    """

    nrow = 5000
    ncol = 5000

    h_min = df_conus_state_basic_info['h_min'].values[i_state]
    h_max = df_conus_state_basic_info['h_max'].values[i_state]
    v_min = df_conus_state_basic_info['v_min'].values[i_state]
    v_max = df_conus_state_basic_info['v_max'].values[i_state]

    state_name = df_conus_state_basic_info['NAME'].values[i_state]

    list_tile_name = df_conus_state_basic_info['tile_name'].values[i_state].split(';')
    assert len(list_tile_name) > 0, f'{state_name} has no tile'

    # use unit8 format to save memory
    img_state_isp_stack = np.zeros((len(array_year), (v_max - v_min + 1) * nrow, (h_max - h_min + 1) * ncol), dtype=np.uint8)

    for i_tile, tile_name in enumerate(list_tile_name):

        h_index, v_index = find_h_v_index_from_tile_name(tile_name)

        if data_flag == 'conus_isp':
            img_stack_ts_is_pct = load_conus_isp_stack(list_year=array_year,
                                                       tile_name=tile_name,
                                                       isp_folder=isp_folder,
                                                       filename_prefix=filename_prefix,
                                                       rootpath_conus_isp=rootpath_conus_isp)
        elif data_flag == 'annual_nlcd':
            img_stack_ts_is_pct = load_annual_nlcd_isp_stack(list_year=array_year,
                                                             tile_name=tile_name,
                                                             nlcd_folder=nlcd_folder,
                                                             rootpath_nlcd_directory=rootpath_nlcd_directory,
                                                             nlcd_filter_ocean_flag=nlcd_filter_ocean_flag)
        else:
            raise ValueError('The data_flag is not correct')

        img_state_isp_stack[:, (v_index - v_min) * nrow: (v_index - v_min + 1) * nrow,
                            (h_index - h_min) * ncol: (h_index - h_min + 1) * ncol] = img_stack_ts_is_pct

    return img_state_isp_stack


def get_running_task_state_isp_change_stats(df_conus_state_basic_info, list_year, modify_target='state'):
    """
    get the running task

    :param list_year:
    :param list_tile_name:
    :return:
    """

    column_name = f'i_{modify_target}'
    
    list_columns = [column_name, 'year']
    
    df_running_task = pd.DataFrame(columns=list_columns, 
                                   index=np.arange(0, len(df_conus_state_basic_info) * len(list_year)))

    i = 0
    for i_region in range(0, len(df_conus_state_basic_info)):
        for year in list_year:
            df_running_task.loc[i, column_name] = i_region
            df_running_task.loc[i, 'year'] = year
            i += 1

    return df_running_task


@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
    # if __name__ =='__main__':
    # rank = 1
    # n_cores = 100000

    modify_target = 'state'  # state or county
    print(modify_target)

    filename_output = join(rootpath, 'data', 'shapefile', 'CONUS_boundary',
                           f'tl_2023_us_{modify_target}', f'conus_{modify_target}_basic_info.csv')
    df_conus_state_basic_info = pd.read_csv(filename_output)

    # keep Texas which has the largest area
    # df_conus_state_basic_info = df_conus_state_basic_info[df_conus_state_basic_info['STUSPS'] == 'TX']
    # df_conus_state_basic_info = df_conus_state_basic_info.reset_index(drop=True)

    data_flag = 'conus_isp'  # 'conus_isp' or 'annual_nlcd'
    
    rootpath_nlcd_directory = None
    nlcd_folder = 'NLCD_annual'
    nlcd_filter_ocean_flag = False

    # data_flag = 'conus_isp'
    rootpath_conus_isp = None
    
    # isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    # filename_prefix = 'unet_regressor_round_masked_post_processing'
    
    # isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015'
    # filename_prefix = 'unet_regressor_round_masked_post_processing'
    
    isp_folder = 'individual_year_tile_post_processing_is_expansion_ndvi015_sm'
    filename_prefix = 'unet_regressor_round_masked_post_processing'
    
    # isp_folder = 'individual_year_tile_post_processing_mean_ndvi010'
    # filename_prefix = 'unet_regressor_round_masked_post_processing'
    
    # isp_folder = 'individual_year_tile_post_processing_mean_filter'
    # filename_prefix = 'unet_regressor_round_masked_post_processing_mean_filter'

    if data_flag == 'conus_isp':
        array_target_year = np.arange(1985, 2022, 1)
    elif data_flag == 'annual_nlcd':
        array_target_year = np.arange(1985, 2023, 1)

    # array_target_year = np.arange(2000, 2001, 1)

    print(data_flag, isp_folder)

    df_running_task = get_running_task_state_isp_change_stats(df_conus_state_basic_info=df_conus_state_basic_info,
                                                              list_year=array_target_year)

    each_core_block = int(np.ceil(len(df_running_task) / n_cores))
    for i in range(0, each_core_block):
        new_rank = rank - 1 + i * n_cores
        # means that all folder has been processed
        if new_rank > len(df_running_task) - 1:
            print(f'{new_rank} this is the last running task')
        else:

            year = df_running_task.loc[new_rank, 'year']
            i_state = df_running_task.loc[new_rank, 'i_state']

            state_id = df_conus_state_basic_info['id'].values[i_state]
            state_short_name = df_conus_state_basic_info['STUSPS'].values[i_state]
            state_name = df_conus_state_basic_info['NAME'].values[i_state]

            mask_whole_state = extract_whole_region_mask(df_conus_state_basic_info, i_state, nrow=5000, ncol=5000, modify_target=modify_target)

            # read the impervious surface area

            print(f'{state_name} {year}')

            array_year = np.arange(year, year + 2, 1)

            img_state_isp_stack = load_state_isp_stack(df_conus_state_basic_info, i_state,
                                                       array_year=array_year,
                                                       data_flag=data_flag,
                                                       isp_folder=isp_folder,
                                                       filename_prefix=filename_prefix,
                                                       nlcd_folder=nlcd_folder,
                                                       nlcd_filter_ocean_flag=nlcd_filter_ocean_flag,
                                                       rootpath_conus_isp=rootpath_conus_isp,
                                                       rootpath_nlcd_directory=rootpath_nlcd_directory,
                                                       )
            ##
            # set the out-state pixels to 255, i.e., nodata value
            img_state_isp_stack[:, mask_whole_state == False] = 255

            # calculate the ISP change stats
            img_isp_change_stats = get_isp_change_stats(img_state_isp_stack)

            ##
            # save the ISP change stats
            if data_flag == 'conus_isp':
                output_path = join(rootpath, 'results', 'isp_change_stats',
                                   'state_level', 'conus_isp', f'{isp_folder}', f'{state_name}')
                if not exists(output_path):
                    os.makedirs(output_path, exist_ok=True)

            elif data_flag == 'annual_nlcd':
                output_path = join(rootpath, 'results', 'isp_change_stats',
                                   'state_level', 'annual_nlcd', f'{state_name}')
                if not exists(output_path):
                    os.makedirs(output_path, exist_ok=True)
            else:
                raise ValueError('The data_flag is not correct')

            output_filename = join(
                output_path, f'{state_short_name}_isp_change_stats_{array_year[0]}_{array_year[1]}.npy')
            print(output_filename)
            np.save(output_filename, img_isp_change_stats)


if __name__ == '__main__':
    main()
