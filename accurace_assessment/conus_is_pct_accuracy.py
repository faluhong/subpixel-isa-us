"""
    evaluate the CONUS IS percentage accuracy using the digitalized reference sample
"""

import numpy as np
import pandas as pd
import matplotlib
import os
from os.path import join, exists
import sys
import fiona
import time
from osgeo import ogr, gdal, osr, gdalconst, gdal_array
import geopandas as gpd
import matplotlib.pyplot as plt

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = os.path.join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from conus_isp_analysis.utils_load_merge_conus_isp import (load_merge_conus_is, load_merge_conus_annual_nlcd)
from deep_learning_isp.utils_deep_learning import add_pyramids_color_in_nlcd_isp_tif
from evaluation.utils_plot_isp import plot_isp_single
from auxiliary_data_process.utils_get_us_land_boundary import (get_us_proj_information)

from Basic_tools.Error_statistical import Error_statistical
from evaluation.utils_evaluation import plot_validation_scatterplot


def get_evaluation_sample_chips(gpd_sample, sample_block_size, path_sample_folder, sample_folder, output_filename_prefix):
    """
        get the evaluation sample chips for the accuracy assessment

        :param gpd_sample:
        :param sample_block_size:
        :param path_sample_folder:
        :param sample_folder:
        :param output_filename_prefix:
        :return:
    """


    img_reference_isp = np.zeros((len(gpd_sample), sample_block_size, sample_block_size), dtype=float)
    img_annual_nlcd_isp = np.zeros((len(gpd_sample), sample_block_size, sample_block_size), dtype=float)
    img_conus_isp = np.zeros((len(gpd_sample), sample_block_size, sample_block_size), dtype=float)

    for i_sample in range(0, len(gpd_sample)):
    # for i_sample in range(0, 1):
        sample_id = gpd_sample['sample_id'].values[i_sample]
        year = gpd_sample['year'].values[i_sample]

        # print(sample_id, year)

        filename_reference = join(path_sample_folder, 'sample_processing', f'{sample_id:03d}',
                                  f'{sample_folder}_{sample_id:03d}_{year}_is_pct_round.tif')

        filename_annual_nlcd = join(path_sample_folder, 'extract_conus_is_chips', 'annual_nlcd',
                                    f'{sample_id:03d}_annual_nlcd_{year}_is_pct.tif')

        filename_conus_isp = join(path_sample_folder, 'extract_conus_is_chips', output_filename_prefix,
                                  f'{sample_id:03d}_{output_filename_prefix}_{year}_is_pct.tif')

        img_reference_isp[i_sample, :, :] = gdal_array.LoadFile(filename_reference).astype(float)

        img_annual_nlcd_isp[i_sample, :, :] = gdal_array.LoadFile(filename_annual_nlcd).astype(float)

        img_conus_isp[i_sample, :, :] = gdal_array.LoadFile(filename_conus_isp).astype(float)

    return (img_reference_isp, img_annual_nlcd_isp, img_conus_isp)


def get_accuracy_report_different_resolution(img_reference_isp,
                                             img_annual_nlcd_isp,
                                             img_conus_isp,
                                             sample_block_radius,
                                             array_resolution=np.array([30, 90, 150, 210, 270])):
    """
        get the accuracy report for different resolution levels

        :param img_reference_isp:
        :param img_annual_nlcd_isp:
        :param img_conus_isp:
        :param sample_block_radius:
        :param array_resolution: list of resolution levels to evaluate the accuracy
    """

    df_evaluate_accuracy = pd.DataFrame(columns=['data_type', 'resolution', 'N_sample', 'bias (estimation - reference)', 'MAE', 'RMSE', 'R2'],
                                        index=np.arange(0, len(array_resolution) * 2, 1))

    array_reference_isp_sum = np.zeros((len(array_resolution), np.shape(img_reference_isp)[0]), dtype=float)
    array_annual_nlcd_isp_sum = np.zeros((len(array_resolution), np.shape(img_reference_isp)[0]), dtype=float)
    array_conus_isp_sum = np.zeros((len(array_resolution), np.shape(img_reference_isp)[0]), dtype=float)

    for i_resolution in range(0, len(array_resolution)):
        resolution = array_resolution[i_resolution]
        resolution_window = int(((resolution / 30) - 1) / 2)  # resolution window to evaluate the accuracy
        print(f'{resolution}m')

        array_reference_isp = img_reference_isp[:, (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1),
                              (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1)]

        array_reference_isp = np.nanmean(array_reference_isp, axis=(1, 2))
        # print(np.shape(array_reference_isp))

        array_annual_nlcd_isp = img_annual_nlcd_isp[:, (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1),
                                (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1)]
        array_annual_nlcd_isp = np.nanmean(array_annual_nlcd_isp, axis=(1, 2))

        array_conus_isp = img_conus_isp[:, (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1),
                          (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1)]
        array_conus_isp = np.nanmean(array_conus_isp, axis=(1, 2))

        array_reference_isp_sum[i_resolution, :] = array_reference_isp
        array_annual_nlcd_isp_sum[i_resolution, :] = array_annual_nlcd_isp
        array_conus_isp_sum[i_resolution, :] = array_conus_isp

        # get the accuracy assessment statistics
        df_evaluate_accuracy.loc[i_resolution * 2, 'data_type'] = 'annual_nlcd'
        df_evaluate_accuracy.loc[i_resolution * 2, 'resolution'] = f'{resolution}m'
        df_evaluate_accuracy.loc[i_resolution * 2, 'N_sample'] = array_reference_isp.size

        error_stats_annual_nlcd = Error_statistical(array_annual_nlcd_isp, array_reference_isp)
        df_evaluate_accuracy.loc[i_resolution * 2, 'bias (estimation - reference)'] = error_stats_annual_nlcd.Bias
        df_evaluate_accuracy.loc[i_resolution * 2, 'MAE'] = error_stats_annual_nlcd.MAE
        df_evaluate_accuracy.loc[i_resolution * 2, 'RMSE'] = error_stats_annual_nlcd.RMSE
        df_evaluate_accuracy.loc[i_resolution * 2, 'R2'] = error_stats_annual_nlcd.R_square

        df_evaluate_accuracy.loc[i_resolution * 2 + 1, 'data_type'] = 'conus_isp'
        df_evaluate_accuracy.loc[i_resolution * 2 + 1, 'resolution'] = f'{resolution}m'
        df_evaluate_accuracy.loc[i_resolution * 2 + 1, 'N_sample'] = array_reference_isp.size

        error_stats_conus_isp = Error_statistical(array_conus_isp, array_reference_isp)
        df_evaluate_accuracy.loc[i_resolution * 2 + 1, 'bias (estimation - reference)'] = error_stats_conus_isp.Bias
        df_evaluate_accuracy.loc[i_resolution * 2 + 1, 'MAE'] = error_stats_conus_isp.MAE
        df_evaluate_accuracy.loc[i_resolution * 2 + 1, 'RMSE'] = error_stats_conus_isp.RMSE
        df_evaluate_accuracy.loc[i_resolution * 2 + 1, 'R2'] = error_stats_conus_isp.R_square

    return (df_evaluate_accuracy, array_reference_isp_sum, array_annual_nlcd_isp_sum, array_conus_isp_sum)


def get_accuracy_report_sample(img_reference_isp,
                               img_annual_nlcd_isp,
                               img_conus_isp,
                               len_sample,
                               ):
    """
        get the accuracy report for each sample block

        :param img_reference_isp:
        :param img_annual_nlcd_isp:
        :param img_conus_isp:
        :param len_sample:
        :return:
    """
    # get the accuracy report for each sample plot
    df_evaluate_accuracy = pd.DataFrame(columns=['data_type', 'sample_id', 'N_sample', 'bias (estimation - reference)', 'MAE', 'RMSE', 'R2'],
                                        index=np.arange(0, 2 * (len_sample + 1), 1))

    for i_sample in range(0, len_sample):
        array_reference_isp = img_reference_isp[i_sample, :, :]

        array_annual_nlcd_isp = img_annual_nlcd_isp[i_sample, :, :]

        array_conus_isp = img_conus_isp[i_sample, :, :]

        # get the accuracy assessment statistics
        df_evaluate_accuracy.loc[i_sample * 2, 'data_type'] = 'annual_nlcd'
        df_evaluate_accuracy.loc[i_sample * 2, 'sample_id'] = f'{i_sample + 1:03d}'
        df_evaluate_accuracy.loc[i_sample * 2, 'N_sample'] = array_reference_isp.size

        error_stats_annual_nlcd = Error_statistical(array_annual_nlcd_isp, array_reference_isp)
        df_evaluate_accuracy.loc[i_sample * 2, 'bias (estimation - reference)'] = error_stats_annual_nlcd.Bias
        df_evaluate_accuracy.loc[i_sample * 2, 'MAE'] = error_stats_annual_nlcd.MAE
        df_evaluate_accuracy.loc[i_sample * 2, 'RMSE'] = error_stats_annual_nlcd.RMSE
        df_evaluate_accuracy.loc[i_sample * 2, 'R2'] = error_stats_annual_nlcd.R_square

        df_evaluate_accuracy.loc[i_sample * 2 + 1, 'data_type'] = 'conus_isp'
        df_evaluate_accuracy.loc[i_sample * 2 + 1, 'sample_id'] = f'{i_sample + 1:03d}'
        df_evaluate_accuracy.loc[i_sample * 2 + 1, 'N_sample'] = array_reference_isp.size

        error_stats_conus_isp = Error_statistical(array_conus_isp, array_reference_isp)
        df_evaluate_accuracy.loc[i_sample * 2 + 1, 'bias (estimation - reference)'] = error_stats_conus_isp.Bias
        df_evaluate_accuracy.loc[i_sample * 2 + 1, 'MAE'] = error_stats_conus_isp.MAE
        df_evaluate_accuracy.loc[i_sample * 2 + 1, 'RMSE'] = error_stats_conus_isp.RMSE
        df_evaluate_accuracy.loc[i_sample * 2 + 1, 'R2'] = error_stats_conus_isp.R_square

    error_stats_annual_nlcd_all = Error_statistical(img_annual_nlcd_isp, img_reference_isp)
    error_stats_conus_isp_all = Error_statistical(img_conus_isp, img_reference_isp)

    df_evaluate_accuracy.loc[len_sample * 2, 'data_type'] = 'annual_nlcd'
    df_evaluate_accuracy.loc[len_sample * 2, 'sample_id'] = 'all'
    df_evaluate_accuracy.loc[len_sample * 2, 'N_sample'] = img_reference_isp.size
    df_evaluate_accuracy.loc[len_sample * 2, 'bias (estimation - reference)'] = error_stats_annual_nlcd_all.Bias
    df_evaluate_accuracy.loc[len_sample * 2, 'MAE'] = error_stats_annual_nlcd_all.MAE
    df_evaluate_accuracy.loc[len_sample * 2, 'RMSE'] = error_stats_annual_nlcd_all.RMSE
    df_evaluate_accuracy.loc[len_sample * 2, 'R2'] = error_stats_annual_nlcd_all.R_square

    df_evaluate_accuracy.loc[len_sample * 2 + 1, 'data_type'] = 'conus_isp'
    df_evaluate_accuracy.loc[len_sample * 2 + 1, 'sample_id'] = 'all'
    df_evaluate_accuracy.loc[len_sample * 2 + 1, 'N_sample'] = img_reference_isp.size
    df_evaluate_accuracy.loc[len_sample * 2 + 1, 'bias (estimation - reference)'] = error_stats_conus_isp_all.Bias
    df_evaluate_accuracy.loc[len_sample * 2 + 1, 'MAE'] = error_stats_conus_isp_all.MAE
    df_evaluate_accuracy.loc[len_sample * 2 + 1, 'RMSE'] = error_stats_conus_isp_all.RMSE
    df_evaluate_accuracy.loc[len_sample * 2 + 1, 'R2'] = error_stats_conus_isp_all.R_square

    return df_evaluate_accuracy


# def main():
if __name__ == '__main__':

    sample_folder = 'v4_conus_ic_pct_2010_2020'
    sample_block_size = 9
    sample_block_radius = int((sample_block_size - 1) / 2)

    path_sample_folder = join(rootpath, 'results', 'accuracy_assessment', 'conus_is_pct', sample_folder)

    # read the gpkg file for the sample
    output_filename_gpkg = join(path_sample_folder, f'{sample_folder}_is_pct_sample.gpkg')
    gpd_sample = gpd.read_file(output_filename_gpkg, layer=f'{sample_folder}_sample')

    output_filename_prefix = 'conus_isp_post_processing_binary_is_ndvi015_sm'

    ##
    (img_reference_isp,
     img_annual_nlcd_isp,
     img_conus_isp) = get_evaluation_sample_chips(gpd_sample=gpd_sample,
                                                  sample_block_size=sample_block_size,
                                                  path_sample_folder=path_sample_folder,
                                                  sample_folder=sample_folder,
                                                  output_filename_prefix=output_filename_prefix)

    # get accuracy report for different resolution levels, but still the same sample size

    array_resolution = np.array([30, 90, 150, 210, 270])  # array of resolution levels to evaluate the accuracy

    (df_accuracy_resolution,
     array_reference_isp_sum,
     array_annual_nlcd_isp_sum,
     array_conus_isp_sum) = get_accuracy_report_different_resolution(img_reference_isp,
                                                                     img_annual_nlcd_isp,
                                                                     img_conus_isp,
                                                                     sample_block_radius,
                                                                     array_resolution=array_resolution)

    # get the accuracy report for each sample block
    df_accuracy_sample_block = get_accuracy_report_sample(img_reference_isp,
                                                          img_annual_nlcd_isp,
                                                          img_conus_isp,
                                                          len_sample=len(gpd_sample),
                                                          )

    ##
    # df_accuracy_resolution.to_excel(join(path_sample_folder, f'{sample_folder}_isp_accuracy_resolution_output.xlsx'), index=False)
    # df_accuracy_sample_block.to_excel(join(path_sample_folder, f'{sample_folder}_isp_accuracy_sample_block_output.xlsx'), index=False)

    ##
    # plot the ISP figure for each sample block
    # for i_sample in range(0, len(gpd_sample)):
    # # for i_sample in range(0, 1):
    #
    #     figsize = (24, 12)
    #     flag_cbar = True
    #     fig, axes = plt.subplots(ncols=3, nrows=1, figsize=figsize)
    #
    #     for i_ax in range(0, 3):
    #
    #         if i_ax == 0:
    #             plot_isp_single(img_reference_isp[i_sample], title=f'{i_sample+1:03d} Reference ISP', ax_plot=axes[i_ax], flag_cbar=flag_cbar)
    #         elif i_ax == 1:
    #             plot_isp_single(img_conus_isp[i_sample], title=f'{i_sample+1:03d} CONUS ISP', ax_plot=axes[i_ax], flag_cbar=flag_cbar)
    #         elif i_ax == 2:
    #             plot_isp_single(img_annual_nlcd_isp[i_sample], title=f'{i_sample+1:03d} Annual NLCD', ax_plot=axes[i_ax], flag_cbar=flag_cbar)
    #
    #     fig_output_folder = join(path_sample_folder, 'sample_isp_figure')
    #     if not exists(fig_output_folder):
    #         os.makedirs(fig_output_folder, exist_ok=True)
    #
    #     fig.savefig(join(fig_output_folder, f'{i_sample+1:03d}_isp.jpg'), dpi=300, bbox_inches='tight')
    #     plt.close()

    ##
    plot_validation_scatterplot(x=img_annual_nlcd_isp.flatten(),
                                y=img_reference_isp.flatten(),
                                x_label='Annual NLCD ISP',
                                y_label='Reference',
                                cbar_label='Log count',
                                title='Annual NLCD ISP evaluation',
                                errors_stats_plot=Error_statistical(img_annual_nlcd_isp.flatten(), img_reference_isp.flatten()),
                                figsize=(14, 10), xlim=(-2, 102), ylim=(-2, 102),
                                flag_1_to_1=True, flag_reg_line=True,
                                bins='log',
                                gridsize=50)

    plot_validation_scatterplot(x=img_conus_isp.flatten(),
                                y=img_reference_isp.flatten(),
                                x_label='CONUS ISP',
                                y_label='Reference',
                                cbar_label='Log count',
                                title='CONUS ISP evaluation',
                                errors_stats_plot=Error_statistical(img_conus_isp.flatten(), img_reference_isp.flatten()),
                                figsize=(14, 10), xlim=(-2, 102), ylim=(-2, 102),
                                flag_1_to_1=True, flag_reg_line=True,
                                bins='log',
                                gridsize=50)


    ## plot the scatterplot for different resolution levels
    list_resolution = [30, 90, 150, 210, 270]  # list of resolution levels to evaluate the accuracy

    # for i_resolution in range(0, len(list_resolution)):
    for i_resolution in range(0, 1):

        resolution = list_resolution[i_resolution]
        resolution_window = int(((resolution / 30) - 1) / 2)  # resolution window to evaluate the accuracy
        print(f'{resolution}m')

        array_reference_isp = img_reference_isp[:, (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1),
                              (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1)]

        array_reference_isp = np.nanmean(array_reference_isp, axis=(1, 2))

        array_annual_nlcd_isp = img_annual_nlcd_isp[:, (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1),
                                (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1)]
        array_annual_nlcd_isp = np.nanmean(array_annual_nlcd_isp, axis=(1, 2))

        array_conus_isp = img_conus_isp[:, (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1),
                          (sample_block_radius - resolution_window):(sample_block_radius + resolution_window + 1)]
        array_conus_isp = np.nanmean(array_conus_isp, axis=(1, 2))

        plot_validation_scatterplot(x=array_annual_nlcd_isp,
                                    y=array_reference_isp,
                                    x_label='Annual NLCD ISP',
                                    y_label='Reference',
                                    cbar_label='Count',
                                    title=f'Annual NLCD ISP evaluation at {resolution}m',
                                    errors_stats_plot=Error_statistical(img_annual_nlcd_isp.flatten(), img_reference_isp.flatten()),
                                    figsize=(12, 9), xlim=(-2, 102), ylim=(-2, 102),
                                    flag_1_to_1=True, flag_reg_line=False,
                                    bins=None,
                                    gridsize=50)

        plot_validation_scatterplot(x=array_conus_isp,
                                    y=array_reference_isp,
                                    x_label='CONUS ISP',
                                    y_label='Reference',
                                    cbar_label='Count',
                                    title=f'CONUS ISP evaluation at {resolution}m',
                                    errors_stats_plot=Error_statistical(img_conus_isp.flatten(), img_reference_isp.flatten()),
                                    figsize=(12, 9), xlim=(-2, 102), ylim=(-2, 102),
                                    flag_1_to_1=True, flag_reg_line=False,
                                    bins=None,
                                    gridsize=50)







