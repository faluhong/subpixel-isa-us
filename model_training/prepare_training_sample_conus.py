"""
    Prepare training samples for the deep learning model across the CONUS region

    The training data comes from: (1) 2022 Chesapeake Bay ISP; (2) CCAP ISP; (3) EnviroAtlas ISP
"""

import numpy as np
import time
import sys
import os
from os.path import join, exists
import time
import random
from tqdm import tqdm
import glob
import click
from osgeo import gdal, gdal_array, gdalconst
import pandas as pd

pwd = os.getcwd()
rootpath = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from deep_learning_isp.utils_deep_learning import (read_cold_variable,
                                                   predictor_normalize,
                                                   read_topography_data,
                                                   topography_normalize)
from Basic_tools.Figure_plot import FP, FP_ISP

from evaluation.utils_evaluation import convert_8_tile_names_to_6_tile_names, convert_6_tile_names_to_8_tile_names


def get_training_data_output_path_both(output_folder):
    """
        get the training data output path

        :param output_folder: the output folder name

        :return:
            path_x_output: x_training_topography, including the COLD spectral features and topography features
            path_y_output, ISP value used for regression task
            path_y_output_binary: binary ISP label for binary classification task
    """
    path_x_output = join(rootpath, 'results', 'deep_learning', output_folder, 'x_training_topography')
    if not exists(path_x_output):
        os.makedirs(path_x_output, exist_ok=True)

    path_y_output = join(rootpath, 'results', 'deep_learning', output_folder, 'y_label')
    if not exists(path_y_output):
        os.makedirs(path_y_output, exist_ok=True)

    path_y_output_binary = join(rootpath, 'results', 'deep_learning', output_folder, 'y_label_binary')
    if not exists(path_y_output_binary):
        os.makedirs(path_y_output_binary, exist_ok=True)

    return path_x_output, path_y_output, path_y_output_binary


if __name__ == '__main__':

    training_sample_output_folder = 'training_sample_conus_v2'
    norm_boundary_folder = 'maximum_minimum_ref_conus'
    predictor_variable_folder = 'predictor_variable_high_res_isp_training'
    central_reflectance_flag = 'change' # 'change' or 'stable'

    nrow, ncol = 5000, 5000
    chip_size = 256
    n_features = 35

    df_summary_high_res_isp_all = pd.read_excel(join(rootpath, 'data', 'ISP_from_high_res_lc', 'summary_high_resolution_isp.xlsx'))

    list_source = ['2022_Chesapeake_Bay', 'EnviroAtlas', 'vermont']  # '2022_Chesapeake_Bay', 'CCAP', 'EnviroAtlas', 'urban_watch', 'vermont'
    df_matched_high_res_isp = df_summary_high_res_isp_all[df_summary_high_res_isp_all['source'].isin(list_source)]

    # list_tile_name = ['h003v002', 'h003v012', 'h007v013', 'h016v014', 'h021v007',
    #                   'h020v016', 'h025v017', 'h027v008', 'h027v009',
    #                   'h028v008', 'h028v009',
    #                   'h029v004',
    #                   'h030v006']

    # list_evaluation_tile_match = convert_8_tile_names_to_6_tile_names(list_tile_name)
    # df_summary_high_res_isp_training = df_summary_high_res_isp_training[df_summary_high_res_isp_training['tile_id'].isin(list_evaluation_tile_match)]

    index_total_sample = 0

    for i in range(0, len(df_matched_high_res_isp)):
        # for i in range(0, 10):

        source = df_matched_high_res_isp['source'].values[i]
        city_folder = df_matched_high_res_isp['city_folder'].values[i]
        tile_name = df_matched_high_res_isp['tile_id'].values[i]
        tile_name = convert_6_tile_names_to_8_tile_names(tile_name)
        year = df_matched_high_res_isp['year'].values[i]
        filename_high_res_isp = df_matched_high_res_isp['file_name'].values[i]

        # print(source, city_folder, tile_name, year)

        img_cold_feature = read_cold_variable(predictor_variable_folder=predictor_variable_folder,
                                              tile_name=tile_name,
                                              year=year,
                                              central_reflectance_flag=central_reflectance_flag,
                                              rootpath_project_folder=None)

        img_dem, img_slope, img_aspect = read_topography_data(tile_name)

        ##
        img_isp_high_res = gdal_array.LoadFile(filename_high_res_isp)
        img_isp_high_res = img_isp_high_res.astype(float)
        img_isp_high_res[img_isp_high_res == 255] = np.nan

        if np.isnan(img_isp_high_res).all():
            print(f'{tile_name} {year} all nan')
            continue

        # Find the indices where NaN values are located
        nan_indices = np.argwhere(~np.isnan(img_isp_high_res))

        # Get the boundaries of NaN values to narrow down the random sampling range
        min_row, min_col = nan_indices.min(axis=0)
        max_row, max_col = nan_indices.max(axis=0)

        ##
        # determine the sample number per tile, based on the valid pixel number in the high resolution ISP divided by the chip counts
        sample_number_per_tile = int(np.count_nonzero(~np.isnan(img_isp_high_res)) // (chip_size ** 2))
        print(f'{i}, {source} {city_folder} {tile_name} {year} sample number per tile: {sample_number_per_tile}')
        if sample_number_per_tile == 0:
            # if the count of valid sample cannot meet the count of one chip, then skip this tile
            # print(f'tile {tile_name} year {year} has no valid pixel')
            continue

        # determine the times of trying to get the valid sample, based on the sample number per tile,
        # empirically, 100 times of the estimated sample number per tile
        total_times_per_tile_trying = int(sample_number_per_tile * 100)

        index_per_tile = 0  # index of sample per tile
        index_test_times_per_tile = 0 # test times per tile
        while True:
            random_row_id = random.randint(min_row, max_row)
            random_col_id = random.randint(min_col, max_col)

            img_cold_feature_chip = img_cold_feature[:, random_row_id:random_row_id + chip_size, random_col_id:random_col_id + chip_size]
            img_y_label = img_isp_high_res[random_row_id:random_row_id+chip_size, random_col_id:random_col_id+chip_size]

            img_dem_chip = img_dem[random_row_id:random_row_id+chip_size, random_col_id:random_col_id+chip_size]
            img_slope_chip = img_slope[random_row_id:random_row_id+chip_size, random_col_id:random_col_id+chip_size]
            img_aspect_chip = img_aspect[random_row_id:random_row_id+chip_size, random_col_id:random_col_id+chip_size]

            if (img_cold_feature_chip.shape == (n_features, chip_size, chip_size)) & (img_y_label.shape == (chip_size, chip_size)):

                nan_mask = np.nansum(img_cold_feature_chip == 0, axis=0)
                nan_mask = nan_mask == np.shape(img_cold_feature)[0]    # all zero values in the training data means nan values

                if (nan_mask.any() | np.isnan(img_y_label)).any():
                    # Nan values in the training data or label

                    # print(f'row {random_row_id:04d} col {random_col_id:04d} contains nan value')
                    # print(f'nan in training data: {np.sum(nan_mask)}')
                    # print(f'nan in label: {np.sum(np.isnan(img_y_label))}')
                    index_test_times_per_tile += 1
                else:

                    img_y_label_binary = np.where(img_y_label > 0, 1, 0)    # convert the continuous label to binary label

                    path_x_output, path_y_output, path_y_output_binary = get_training_data_output_path_both(output_folder=training_sample_output_folder)

                    filename_x_output = join(path_x_output, f'{(index_total_sample + 1):05d}_{source}_{tile_name}_{year}_{random_row_id:04d}_{random_col_id:04d}_x_training_topography.npy')
                    filename_y_output = join(path_y_output, f'{(index_total_sample + 1):05d}_{source}_{tile_name}_{year}_{random_row_id:04d}_{random_col_id:04d}_y_label.npy')
                    filename_y_output_binary = join(path_y_output_binary, f'{(index_total_sample + 1):05d}_{source}_{tile_name}_{year}_{random_row_id:04d}_{random_col_id:04d}_y_label_binary.npy')

                    # normalize the training data
                    img_cold_feature_chip_normalized = predictor_normalize(img_cold_feature_chip, norm_boundary_folder=norm_boundary_folder)
                    (img_dem_chip_norm, img_slope_chip_norm, img_aspect_chip_norm) = topography_normalize(img_dem_chip, img_slope_chip, img_aspect_chip,
                                                                                                          norm_boundary_folder=norm_boundary_folder)

                    img_x_training_topography = np.concatenate((img_cold_feature_chip_normalized, img_dem_chip_norm, img_slope_chip_norm, img_aspect_chip_norm), axis=0)

                    np.save(filename_x_output, img_x_training_topography)
                    np.save(filename_y_output, img_y_label)
                    np.save(filename_y_output_binary, img_y_label_binary)

                    print(index_total_sample)
                    print(filename_x_output)
                    print(filename_y_output)
                    print(filename_y_output_binary)

                    index_per_tile += 1
                    index_total_sample += 1
                    index_test_times_per_tile += 1

                print(f'index_per_tile: {index_per_tile}, index_test_times_per_tile: {index_test_times_per_tile}, index_total_sample: {index_total_sample}')

                if index_per_tile >= sample_number_per_tile:
                    break

                # after testing for the maximum trying times, if still cannot get the valid sample, then break to avoid the infinite loop
                if index_test_times_per_tile >= total_times_per_tile_trying:
                    print(f'cannot find the valid sample for tile {tile_name} year {year}')
                    break
