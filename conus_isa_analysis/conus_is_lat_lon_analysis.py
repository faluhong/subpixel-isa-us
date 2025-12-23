"""
    analyze the latitude and longitude distribution of ISPs in the CONUS region
"""

import numpy as np
from os.path import join, exists
import os
import sys
import pandas as pd
from osgeo import gdal, gdal_array
import matplotlib.pyplot as plt
import click

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = os.path.join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from conus_isp_analysis.utils_load_merge_conus_isp import load_merge_conus_is


def create_dataframe(array_is_pct_lat, array_is_pct_lon, steps,
                     first_valid_row,
                     # last_valid_row,
                     first_valid_col,
                     # last_valid_col,
                     ):
    """
        Create a DataFrame to store the latitude and longitude ISP percentage data.
        :param array_is_pct_lat:
        :param array_is_pct_lon:
        :param steps:
        :return:
    """

    rows = []

    for i in range(array_is_pct_lat.shape[0]):

        rows.append({
            'direction': 'latitude',
            'range_index': f'{(first_valid_row + i * steps):05d}-{(first_valid_row + (i + 1) * steps):05d}',
            'is_pct': array_is_pct_lat[i]
        })

    for i in range(array_is_pct_lon.shape[0]):
        rows.append({
            'direction': 'longitude',
            'range_index': f'{(first_valid_col + i * steps):05d}-{(first_valid_col + (i + 1) * steps):05d}',
            'is_pct': array_is_pct_lon[i]
        })

    df_is_pct_lat_lon = pd.DataFrame(rows, columns=['direction', 'range_index', 'is_pct'])

    return df_is_pct_lat_lon


def latitude_is_pct_analysis(img_conus_isp, steps,
                             first_valid_row,
                             last_valid_row,
                             first_valid_col,
                             last_valid_col,):
    """
        Analyze the latitude distribution of ISPs in the CONUS region
        :return:
    """

    nrows = last_valid_row - first_valid_row + 1
    ncols = last_valid_col - first_valid_col + 1

    array_is_pct_lat = np.zeros((nrows // steps + 1))

    for i_lat in range(0, nrows, steps):
        array_tmp_calculation = img_conus_isp[i_lat + first_valid_row: i_lat + first_valid_row + steps, :].copy()
        array_tmp_calculation = array_tmp_calculation.astype('float32')
        array_tmp_calculation[array_tmp_calculation == 255] = np.nan  # set no data to nan

        if np.count_nonzero(~np.isnan(array_tmp_calculation)) <= 1000:
            array_is_pct_lat[i_lat // steps] = np.nan
            print(f'{i_lat+first_valid_row:05d}-{i_lat + first_valid_row + steps:05d} {np.nanmean(array_tmp_calculation)}')

        else:
            array_is_pct_lat[i_lat // steps] = np.nanmean(array_tmp_calculation)
            print(f'{i_lat+first_valid_row:05d}-{i_lat + first_valid_row + steps:05d} {np.nanmean(array_tmp_calculation)}')

    return array_is_pct_lat


def longitude_is_pct_analysis(img_conus_isp, steps,
                              first_valid_row,
                              last_valid_row,
                              first_valid_col,
                              last_valid_col,
                              ):
    """
        Analyze the longitude distribution of ISPs in the CONUS region
        :param img_conus_isp:
        :param steps:
        :return:
    """

    nrows = last_valid_row - first_valid_row + 1
    ncols = last_valid_col - first_valid_col + 1

    array_is_pct_lon = np.zeros((ncols // steps + 1))

    for i_lon in range(0, ncols, steps):
        array_tmp_calculation = img_conus_isp[:, i_lon + first_valid_col:i_lon + first_valid_col + steps].copy()
        array_tmp_calculation = array_tmp_calculation.astype('float32')
        array_tmp_calculation[array_tmp_calculation == 255] = np.nan  # set no data to nan

        if np.count_nonzero(~np.isnan(array_tmp_calculation)) <= 1000:
            array_is_pct_lon[i_lon // steps] = np.nan
            print(f'{i_lon + first_valid_col:05d}-{i_lon + first_valid_col + steps:05d} {np.nanmean(array_tmp_calculation)}')

        else:
            array_is_pct_lon[i_lon // steps] = np.nanmean(array_tmp_calculation)
            print(f'{i_lon + first_valid_col:05d}-{i_lon + first_valid_col + steps:05d} {np.nanmean(array_tmp_calculation)}')

        print(i_lon, np.nanmean(array_tmp_calculation))

        array_is_pct_lon[i_lon // steps] = np.nanmean(array_tmp_calculation)

    return array_is_pct_lon


def get_first_last_valid_row_col(img_conus_isp):
    """
        get the first and last valid row and column in the image
        :param img_conus_isp:
        :return:
    """

    # get the first row that has valid data
    first_valid_row = None
    for i in range(img_conus_isp.shape[0]):
        if np.any(img_conus_isp[i, :] != 255):
            first_valid_row = i
            break
    # get the last row that has valid data
    last_valid_row = None
    for i in range(img_conus_isp.shape[0] - 1, -1, -1):
        if np.any(img_conus_isp[i, :] != 255):
            last_valid_row = i
            break

    # get the first column that has valid data
    first_valid_col = None
    for j in range(img_conus_isp.shape[1]):
        if np.any(img_conus_isp[:, j] != 255):
            first_valid_col = j
            break
    # get the last column that has valid data
    last_valid_col = None
    for j in range(img_conus_isp.shape[1] - 1, -1, -1):
        if np.any(img_conus_isp[:, j] != 255):
            last_valid_col = j
            break

    # 4579 101857 6800 160978
    print(first_valid_row, last_valid_row, first_valid_col, last_valid_col)

    return (first_valid_row, last_valid_row, first_valid_col, last_valid_col)


@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
# if __name__ =='__main__':
    # rank = 1
    # n_cores = 10000

    output_folder_merged_conus_isp = 'merge_conus_isp_post_processing_binary_is_ndvi015_sm'  # folder to store the merged conus isp
    output_filename_prefix = 'conus_isp_post_processing_binary_is_ndvi015_sm'

    steps = 3000    # latitude/longitude steps in pixels, each pixel is 30m, 1000 pixels = 30 km, 500 pixels = 15 km

    array_year_to_process = np.arange(1988, 2021, 1)
    
    first_valid_row = 4579
    last_valid_row = 101857
    first_valid_col = 6800
    last_valid_col = 160978
    
    each_core_block = int(np.ceil(len(array_year_to_process) / n_cores))
    for i in range(0, each_core_block):
        new_rank = rank - 1 + i * n_cores
        # means that all folder has been processed
        if new_rank > len(array_year_to_process) - 1:
            print(f'{new_rank} this is the last running task')
        else:
            year = array_year_to_process[new_rank]
            print(f'Processing year: {year}')
    
            img_conus_isp = load_merge_conus_is(output_folder_merged_conus_isp, output_filename_prefix, year, data_type='isp',)

            array_is_pct_lat = latitude_is_pct_analysis(img_conus_isp, steps,
                                                        first_valid_row=first_valid_row,
                                                        last_valid_row=last_valid_row,
                                                        first_valid_col=first_valid_col,
                                                        last_valid_col=last_valid_col,)

            array_is_pct_lon = longitude_is_pct_analysis(img_conus_isp, steps,
                                                         first_valid_row=first_valid_row,
                                                         last_valid_row=last_valid_row,
                                                         first_valid_col=first_valid_col,
                                                         last_valid_col=last_valid_col,)

            df_is_pct_lat_lon = create_dataframe(array_is_pct_lat, array_is_pct_lon, steps,
                                                 first_valid_row=first_valid_row,
                                                 # last_valid_row=last_valid_row,
                                                 first_valid_col=first_valid_col,
                                                 # last_valid_col=last_valid_col,
                                                 )

            df_is_pct_lat_lon = create_dataframe(array_is_pct_lat, array_is_pct_lon, steps,
                                                 first_valid_row=first_valid_row,
                                                 # last_valid_row=last_valid_row,
                                                 first_valid_col=first_valid_col,
                                                 # last_valid_col=last_valid_col,
                                                 )
            ## save to xlsx
            output_folder_analysis = join(rootpath, 'results', 'conus_is_lat_lon', output_folder_merged_conus_isp)
            if not exists(output_folder_analysis):
                os.makedirs(output_folder_analysis, exist_ok=True)

            output_filename_excel = join(output_folder_analysis, f'conus_isp_lat_lon_distribution_{year}_step_{steps}.xlsx')
            df_is_pct_lat_lon.to_excel(output_filename_excel, index=False)


if __name__ == '__main__':
    main()



