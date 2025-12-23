"""
    calculate the impervious surface area for each MSA (metropolitan statistical area) from 1985 to 2022/2023
    
    Using 32 GBs is OK to run the code
     
    Using 50 cores for parallel computing, it takes around 1.0 hours to finish the calculation
    
    The parallel unit is the MSA for the whole period (1985-2022/2023) 
    The output is the ISP change statistics for each MSA regions from 1985 to 2022/2023.
    
    The final output size is about 3 GBs.
    
    The maximum cores that HPC priority partition can provide is around 260
"""

from itertools import count
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

from auxiliary_data_process.prepare_state_county_basic_info import (extract_whole_region_mask)
from Basic_tools.Figure_plot import FP
from evaluation.utils_plot_isp import plot_isp_single
from conus_isp_analysis.tile_is_change_stats_output import get_isp_change_stats

from conus_isp_analysis.state_is_area_cal import (load_state_isp_stack, get_running_task_state_isp_change_stats)


@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
# if __name__ =='__main__':
    # rank = 1
    # n_cores = 1

    modify_target = 'msa'  # state, county, or msa
    print(modify_target)

    path_msa_2015 = join(rootpath, 'data', 'urban_pulse', 'shapefile', 'cb_2015_us_cbsa_500k')
    
    filename_output = join(path_msa_2015, 'cb_2015_us_cbsa_500k_ard_conus.csv')
    df_conus_state_basic_info = pd.read_csv(filename_output)

    data_flag = 'annual_nlcd'  # 'conus_isp' or 'annual_nlcd'
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

    print(data_flag, isp_folder)

    df_running_task = df_conus_state_basic_info.copy()
    
    each_core_block = int(np.ceil(len(df_running_task) / n_cores))
    for i in range(0, each_core_block):
        new_rank = rank - 1 + i * n_cores
        # means that all folder has been processed
        if new_rank > len(df_running_task) - 1:
            print(f'{new_rank} this is the last running task')
        else:
            
            img_sum_isp_change_stats = np.zeros((len(array_target_year), 101, 101), dtype=int)
            
            i_state = new_rank
            
            msa_name = df_conus_state_basic_info['NAME'].values[i_state]

            # replace the '/' with '_' to avoid the error in the file name, such as Louisville/Jefferson County, KY-IN
            msa_name = msa_name.replace(r'/', '_')  
            
            mask_whole_county = extract_whole_region_mask(df_conus_state_basic_info, i_state, 
                                                          nrow=5000, ncol=5000,
                                                          modify_target=modify_target)
            
            # save the ISP change stats
            if data_flag == 'conus_isp':
                output_path = join(rootpath, 'results', 'isp_change_stats', f'{modify_target}_level', data_flag,  f'{isp_folder}')

            elif data_flag == 'annual_nlcd':
                output_path = join(rootpath, 'results', 'isp_change_stats', f'{modify_target}_level', data_flag)
                
            else:
                raise ValueError('The data_flag is not correct')

            if not exists(output_path):
                os.makedirs(output_path, exist_ok=True)

            output_filename = join(output_path, f'{msa_name}_isp_change_stats_{array_target_year[0]}_{array_target_year[-1]}.npy')
            # print(output_filename)
            
            if exists(output_filename):
                print(f'exists {output_filename}')
                # no need to run the code if the output file exists
                pass
            else:
                
                print(f'running needed {output_filename}')
                
                for i_year, year in enumerate(array_target_year):

                    print(f'{msa_name} {year}')

                    # read the impervious surface area
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
                    
                    # set the out-state pixels to 255, i.e., nodata value
                    img_state_isp_stack[:, mask_whole_county == False] = 255

                    # calculate the ISP change stats
                    img_isp_change_stats = get_isp_change_stats(img_state_isp_stack)
                    
                    # save the ISP change stats in a single year to the img_sum_isp_change_stats
                    img_sum_isp_change_stats[i_year, :, :] = img_isp_change_stats

                np.save(output_filename, img_sum_isp_change_stats)


if __name__ == '__main__':
    main()
