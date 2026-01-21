"""
    calculate the annual centroid movement distance for each target (state, MSA)
"""

import numpy as np
import os
from os.path import join, exists
import sys
import geopandas as gpd

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


from pythoncode.conus_isa_centroid.is_centroid_extraction import (get_target_basic_info)
from pythoncode.conus_isa_centroid.utils_centroid_trajectory_stats import (calculate_annual_centroid_distance_move)


def read_centroid_gpkg(modify_target, data_flag, isp_folder,):
    """
        read the centroid gpkg file
    """

    if modify_target == 'conus':
        output_merge_centroid_gpkg_filename = join(rootpath, 'results', 'is_centroid', f'{modify_target}_level', data_flag, isp_folder,
                                                   f'conus_centroid.gpkg')
    else:
        output_merge_centroid_gpkg_filename = join(rootpath, 'results', 'is_centroid', f'{modify_target}_level', data_flag, isp_folder,
                                                   f'conus_is_centroid_all_{modify_target}.gpkg')

    gpd_centroid = gpd.read_file(output_merge_centroid_gpkg_filename)

    return gpd_centroid


def get_list_unique_name(modify_target, gpd_centroid):

    # state, MSA, county level
    if modify_target == 'county':
        # some states may have the same count name, so we need to use a unique name based on the state and county name
        gpd_centroid['UNIQUE_NAME'] = gpd_centroid['STATE_STUSPS'].astype(str) + '_' + gpd_centroid['NAMELSAD'].astype(str)
    else:
        # for state and MSA level, we can use the name directly
        gpd_centroid['UNIQUE_NAME'] = gpd_centroid['NAME'].astype(str)

    list_unique_name = gpd_centroid['UNIQUE_NAME'].unique()

    return list_unique_name


# def main():
if __name__ =='__main__':

    ##
    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    modify_target = 'state'     # 'state', 'msa'

    print(f'Processing {data_flag} with ISP folder: {isp_folder}, modify target: {modify_target}')

    # read the basic info of the target
    df_basic_info = get_target_basic_info(modify_target, flag_keep_geometry=True, flag_wgs84=True, flag_all_msa=False)

    # read the annual centroid GeoPackage file, used to analyze the financial crisis impact on the centroid movement
    gpd_centroid_annual = read_centroid_gpkg(modify_target, data_flag, isp_folder)
    list_unique_name = get_list_unique_name(modify_target, gpd_centroid_annual)

    ##
    # analyze the centroid movement distance per year before and after the financial crisis
    for i_target in range(0, len(list_unique_name)):

        target_name = list_unique_name[i_target]
        print(target_name)

        gpd_centroid_target = gpd_centroid_annual[gpd_centroid_annual['UNIQUE_NAME'] == target_name]

        array_year_plot = gpd_centroid_target['year'].values.astype(int)
        array_latitude = gpd_centroid_target['latitude'].values.astype(float)
        array_longitude = gpd_centroid_target['longitude'].values.astype(float)

        array_annual_distance_m = calculate_annual_centroid_distance_move(array_latitude, array_longitude)

        for year in range(0, len(array_year_plot) - 1):
            print(f'Year: {array_year_plot[year]} to {array_year_plot[year + 1]}, Distance (m): {array_annual_distance_m[year]}')

            df_basic_info.loc[i_target, f'centroid_distance_m_{array_year_plot[year]}_{array_year_plot[year + 1]}'] = array_annual_distance_m[year]

    ##
    # save the basic info with annual centroid movement distance

    # output_folder = join(rootpath, 'results', 'is_centroid', f'{modify_target}_level', data_flag, isp_folder)
    # if not exists(output_folder):
    #     os.makedirs(output_folder, exist_ok=True)
    #
    # df_basic_info.to_file(join(output_folder, f'{modify_target}_annual_centroid_distance.gpkg'), driver='GPKG')
    #
    # df_basic_info_drop_geometry = df_basic_info.drop(columns='geometry')
    # df_basic_info_drop_geometry.to_excel(join(output_folder, f'{modify_target}_annual_centroid_distance.xlsx'), index=False)

    ##




