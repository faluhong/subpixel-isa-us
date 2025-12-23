"""
    extract the IS centroid information from the ISP stack at different levels (state, county, MSA)

    The code is modified to run for each year independently. This can save the memory and easier for parallel processing.
    
    It's recommended to run this code using priority partition. The general partition can have the out-of-memory issue
    
    For the state running, it requires 128 GB memory, especially for the Texas state. 
    Asking for 50 cores running, it takes about 1 hour to finish the all the states.
    
    For the county-level running, it's OK to use 32 GB memory (The requirement is about 16 GB memory, but to be safe, use 32 GB).
    Asking for 50 cores running, it takes about 4 hour to finish the all the counties.
"""

import os
from os.path import join, exists
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import matplotlib
import numpy as np
import seaborn as sns
import geopandas as gpd
import click

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP

from conus_isp_trajectory.is_centroid_analysis import (get_map_year, get_target_basic_info,
                                                       load_isp_admin_boundary_stack,
                                                       get_annual_centroid_loc,
                                                       generate_geodataframe_centroid_info,
                                                       get_centroid_gpkg_output_filename,
                                                       plot_isp_centroid_movement)


@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
@click.option('--modify_target', type=str, default='msa', help='the target to modify, e.g., msa, county or state')
def main(rank, n_cores, modify_target):
# if __name__ == '__main__':

    # rank = 26
    # n_cores = 100000
    # modify_target = 'state' # 'msa', 'county' or 'state'

    # 1345 for Connecticut county in the CONUS basic info dataframe, 10 for Connecticut state, 596 for Connecticut Hartford MSA
    # i_target = 596

    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    filename_prefix = 'unet_regressor_round_masked_post_processing'

    array_year = get_map_year(data_flag)

    df_conus_target_basic_info = get_target_basic_info(modify_target=modify_target)

    each_core_block = int(np.ceil(len(df_conus_target_basic_info) / n_cores))
    for i in range(0, each_core_block):
        new_rank = rank - 1 + i * n_cores
        # means that all folder has been processed
        if new_rank > len(df_conus_target_basic_info) - 1:
            print(f'{new_rank} this is the last running task')
        else:

            # load CT ISP stack, i_target = 10
            # load the CT Northwest Hills county ISP stack, i_target = 1345
            # load the CT Hartford MSA ISP stack, i_target = 596
            
            gdf_centroid_all = gpd.GeoDataFrame()

            for year in array_year:
                print(f'Processing year: {year}')

                (img_isp_stack) = load_isp_admin_boundary_stack(data_flag=data_flag,
                                                                      isp_folder=isp_folder,
                                                                      filename_prefix=filename_prefix,
                                                                      df_conus_target_basic_info=df_conus_target_basic_info,
                                                                      i_target=new_rank,
                                                                      modify_target=modify_target,
                                                                      array_target_year=np.array([year]),)

                # get the annual centroid of the ISP stack
                (array_cy, array_cx) = get_annual_centroid_loc(img_isp_stack)
                print(array_cy, array_cx)
                
                del img_isp_stack  # free memory

                # define the GeoDataFrame to store the centroid information
                gdf_centroid = generate_geodataframe_centroid_info(array_cy=array_cy,
                                                                   array_cx=array_cx,
                                                                   df_conus_state_basic_info=df_conus_target_basic_info,
                                                                   i_target=new_rank,
                                                                   array_year=np.array([year]),)

                # append the GeoDataFrame to the gdf_centroid_all
                gdf_centroid_all = pd.concat([gdf_centroid_all, gdf_centroid], ignore_index=True)

            # output the centroid information to a GeoPackage file
            output_filename = get_centroid_gpkg_output_filename(modify_target=modify_target,
                                                                df_conus_target_basic_info=df_conus_target_basic_info,
                                                                i_target=new_rank,
                                                                data_flag=data_flag,
                                                                isp_folder=isp_folder)

            gdf_centroid.to_file(output_filename, driver='GPKG')
            print(f'Centroid information saved to: {output_filename}')

  
if __name__ == '__main__':
    main()
