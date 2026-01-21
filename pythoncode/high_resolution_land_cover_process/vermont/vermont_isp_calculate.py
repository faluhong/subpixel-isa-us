"""
    calculate the ISP for Vermont
"""

import os
from os.path import join
import sys
from osgeo import gdal_array
import pandas as pd
import glob
import numpy as np
import logging

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.high_resolution_land_cover_process.utils_high_resolution_land_cover_process import output_estimated_isp


def define_logger(path_vermont):
    """
        define the logger for the isp calculation
    :return:
    """
    logger_isp_cal = logging.getLogger('logger_isp_cal')
    logger_isp_cal.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(join(path_vermont, 'vermont_logger_isp_cal.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger_isp_cal.addHandler(file_handler)

    logger_isp_cal.addHandler(file_handler)

    return logger_isp_cal


def vermont_get_running_task_dataframe(path_vermont):
    """
    get the dataframe recording the running task

    :param df_ccap:
    :return:
    """

    df_running_task = pd.DataFrame(columns=['city_name', 'year', 'tile_name', 'file_name'])

    index = 0
    list_file_name_high_res_lc = glob.glob(join(path_vermont, 'clip_ard_high_resolution', '*.tif'))

    for j in range(0, len(list_file_name_high_res_lc)):
        file_name_high_res_lc = list_file_name_high_res_lc[j]
        tile_name = os.path.split(file_name_high_res_lc)[-1][-10:-4]

        df_running_task.loc[index, 'city_name'] = 'vermont'
        df_running_task.loc[index, 'year'] = 2016
        df_running_task.loc[index, 'tile_name'] = tile_name
        df_running_task.loc[index, 'file_name'] = file_name_high_res_lc

        index += 1

    return df_running_task


def vermont_calculate_isp(img_high_res_lc):
    """
    calculate the impervious surface percentage from high resolution land cover

    :param img_high_res_lc:
    :return:
    """

    landsat_pixel_size = 30
    nrow_1m, ncol_1m = 150000, 150000
    nrow_30m, ncol_30m = nrow_1m // landsat_pixel_size, ncol_1m // landsat_pixel_size

    img_isp = np.zeros((nrow_30m, ncol_30m), dtype=np.float32)

    for row_id in range(0, nrow_1m, landsat_pixel_size):

        non_data_pixel_count_row = np.count_nonzero((img_high_res_lc[row_id:row_id + landsat_pixel_size, :] == 0))

        if non_data_pixel_count_row == ncol_1m * landsat_pixel_size:
            img_isp[row_id // landsat_pixel_size : row_id // landsat_pixel_size + 1, :] = np.nan
            # print(row_id, 'no valid data in the row')
        else:
            for col_id in range(0, ncol_1m, landsat_pixel_size):

                # non_data_pixel_count = np.count_nonzero((img_is_mask_1m[row_id:row_id + landsat_pixel_size, col_id:col_id + landsat_pixel_size] == 0))
                non_data_pixel_count = np.count_nonzero((img_high_res_lc[row_id:row_id + landsat_pixel_size, col_id:col_id + landsat_pixel_size] == 0))

                if non_data_pixel_count > 0:
                    img_isp[row_id // landsat_pixel_size, col_id // landsat_pixel_size] = np.nan
                    # print(row_id, col_id, non_data_pixel_count)
                else:

                    lc_block = img_high_res_lc[row_id:row_id + landsat_pixel_size, col_id:col_id + landsat_pixel_size]

                    # 'Buildings', 'Roads', 'Other Impervious', 'Railroads', 'Compacted Bare Soil' are labeled as impervious surface
                    # Compacted Bare Soil usually means mining area
                    isp_count = np.count_nonzero(
                        (lc_block == 5) | (lc_block == 6) | (lc_block == 7) | (lc_block == 8) | (lc_block == 9)
                    )

                    isp_pecentage = isp_count / landsat_pixel_size / landsat_pixel_size * 100
                    img_isp[row_id // landsat_pixel_size, col_id // landsat_pixel_size] = isp_pecentage
                    # print(row_id, col_id, isp_pecentage)

    return img_isp


# @click.command()
# @click.option('--rank', type=int, default=0, help='job array id, e.g., 0-199')
# @click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
if __name__ == '__main__':
# def main(rank, n_cores):

    n_cores = 1
    rank = 1

    path_vermont = join(rootpath, 'data', 'high_resolution_land_cover', 'Vermont')

    logger_isp_cal = define_logger(path_vermont)

    df_running_task = vermont_get_running_task_dataframe(path_vermont)

    ##
    each_core_block = int(np.ceil(len(df_running_task) / n_cores))
    for i in range(0, each_core_block):

        new_rank = rank - 1 + i * n_cores

        if new_rank > len(df_running_task) - 1:  # means that all folder has been processed
            logger_isp_cal.info(f'{new_rank} this is the last running task')
        else:
            folder_name = df_running_task.loc[new_rank, 'city_name']
            year = df_running_task.loc[new_rank, 'year']
            tile_name = df_running_task.loc[new_rank, 'tile_name']
            file_name_high_res_lc = df_running_task.loc[new_rank, 'file_name']

            print(folder_name, year, tile_name)

            logger_isp_cal.info(f'loading the high resolution land cover {folder_name} {year} {tile_name} {file_name_high_res_lc}')
            img_high_res_lc = gdal_array.LoadFile(file_name_high_res_lc)

            logger_isp_cal.info(f'calculating the ISP for {folder_name} {year} {tile_name}')
            img_isp = vermont_calculate_isp(img_high_res_lc)
            logger_isp_cal.info(f'ISP calculation is done for {folder_name} {year} {tile_name}')

            if np.isnan(img_isp).all():
                logger_isp_cal.info(f'no valid data in the image for {folder_name} {year} {tile_name}')
            else:
                output_filename = join(rootpath, 'data', 'ISP_from_high_res_lc', 'vermont', f'{folder_name}_{year}_{tile_name}_ISP.tif')
                logger_isp_cal.info(f'output the ISP for {folder_name} {year} {tile_name} {output_filename}')
                output_filename = output_estimated_isp(img_isp, tile_name, output_filename)



