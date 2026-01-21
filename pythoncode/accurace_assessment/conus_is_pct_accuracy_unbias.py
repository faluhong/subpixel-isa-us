"""
    calculate the unbiased CONUS IS percentage accuracy by incorporating the area/weight
"""

import numpy as np
import pandas as pd
import os
from os.path import join
import sys
import geopandas as gpd

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = os.path.join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.util_function.Error_statistical import Error_statistical
from pythoncode.accurace_assessment.conus_is_pct_accuracy import (get_evaluation_sample_chips,
                                                                  get_accuracy_report_different_resolution)


def prepare_evaluation_ard_data(sample_folder, sample_block_size,
                                array_resolution=np.array([30, 90, 150, 210, 270]),
                                output_filename_prefix='conus_isp_post_processing_binary_is_ndvi015_sm'):
    """
        prepare the evaluation ARD data for the accuracy assessment

        :param sample_folder:
        :param sample_block_size:
        :return:
    """

    sample_block_radius = int((sample_block_size - 1) / 2)

    path_sample_folder = join(rootpath, 'results', 'accuracy_assessment', 'conus_is_pct')

    # read the gpkg file for the sample
    output_filename_gpkg = join(path_sample_folder, f'{sample_folder}_is_pct_sample.gpkg')
    gpd_sample = gpd.read_file(output_filename_gpkg, layer=f'{sample_folder}_sample')

    # get the evaluation sample chips
    (img_reference_isp,
     img_annual_nlcd_isp,
     img_conus_isp) = get_evaluation_sample_chips(gpd_sample=gpd_sample,
                                                  sample_block_size=sample_block_size,
                                                  path_sample_folder=path_sample_folder,
                                                  sample_folder=sample_folder,
                                                  output_filename_prefix=output_filename_prefix)

    (df_accuracy_resolution,
     array_reference_isp_sum,
     array_annual_nlcd_isp_sum,
     array_conus_isp_sum) = get_accuracy_report_different_resolution(img_reference_isp,
                                                                     img_annual_nlcd_isp,
                                                                     img_conus_isp,
                                                                     sample_block_radius,
                                                                     array_resolution=array_resolution)

    return (array_reference_isp_sum, array_annual_nlcd_isp_sum, array_conus_isp_sum)


def get_ns_is_weight(data_flag, isp_folder, array_target_year):
    """
        get the weight of natural surface and impervious surface

        :param data_flag: the data flag, such as 'conus_isp' or 'annual_nlcd'
        :param isp_folder:
        :param array_target_year: the target year of CONUS IPS mapping, such as from 1988 to 2020
        :return:
    """

    if data_flag == 'annual_nlcd':
        output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag)
    elif data_flag == 'conus_isp':
        output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag, isp_folder)
    else:
        raise ValueError('data_flag must be conus_isp or annual_nlcd')

    df_is_change_sum_conus = pd.read_csv(join(output_folder, 'conus_is_change_type_summary.csv'))

    df_is_change_sum_conus_cal = df_is_change_sum_conus[np.isin(df_is_change_sum_conus['year_2'].values, array_target_year)].copy()

    assert len(df_is_change_sum_conus_cal) > 0, 'No data for the selected year range'

    array_count_ns = df_is_change_sum_conus_cal['count_stable_natural'].values + df_is_change_sum_conus_cal['count_is_reversal'].values

    array_count_is = (df_is_change_sum_conus_cal['count_stable_is'].values
                      + df_is_change_sum_conus_cal['count_is_expansion'].values
                      + df_is_change_sum_conus_cal['count_is_intensification'].values
                      + df_is_change_sum_conus_cal['count_is_decline'])

    array_weight_ns = array_count_ns / (array_count_ns + array_count_is)
    array_weight_is = array_count_is / (array_count_ns + array_count_is)

    weight_ns_mean = np.nanmean(array_weight_ns)
    weight_is_mean = np.nanmean(array_weight_is)

    return (weight_ns_mean, weight_is_mean)


def calculate_unbias_accuracy(array_conus_isp, array_reference_isp, weight_ns_mean, weight_is_mean, print_flag=True):
    """
        calculate the unbiased accuracy for the CONUS ISP percentage

        :param array_conus_isp:
        :param array_reference_isp:
        :param weight_ns_mean:
        :param weight_is_mean:
        :return:
    """

    # get the unbiased accuracy
    mask_natural = array_conus_isp == 0

    error_stats_conus_isp_ns = Error_statistical(array_conus_isp[mask_natural], array_reference_isp[mask_natural])
    error_stats_conus_isp_is = Error_statistical(array_conus_isp[~mask_natural], array_reference_isp[~mask_natural])

    if print_flag:
        error_stats_conus_isp_ns.print_error_stats()
        error_stats_conus_isp_is.print_error_stats()

    if (mask_natural == True).all():
        # only natural surface, no impervious surface
        bias_weight = weight_ns_mean * error_stats_conus_isp_ns.Bias
        mae_weight = weight_ns_mean * error_stats_conus_isp_ns.MAE
        rmse_weight = np.sqrt(weight_ns_mean * error_stats_conus_isp_ns.RMSE * error_stats_conus_isp_ns.RMSE)

    elif (mask_natural == False).all():
        # only impervious surface, no natural surface
        bias_weight = weight_is_mean * error_stats_conus_isp_is.Bias
        mae_weight = weight_is_mean * error_stats_conus_isp_is.MAE
        rmse_weight = np.sqrt(weight_is_mean * error_stats_conus_isp_is.RMSE * error_stats_conus_isp_is.RMSE)

    else:
        bias_weight = weight_ns_mean * error_stats_conus_isp_ns.Bias + weight_is_mean * error_stats_conus_isp_is.Bias
        mae_weight = weight_ns_mean * error_stats_conus_isp_ns.MAE + weight_is_mean * error_stats_conus_isp_is.MAE

        rmse_weight = np.sqrt(weight_ns_mean * error_stats_conus_isp_ns.RMSE * error_stats_conus_isp_ns.RMSE
                              + weight_is_mean * error_stats_conus_isp_is.RMSE * error_stats_conus_isp_is.RMSE)

    # print(bias_weight, mae_weight, rmse_weight)
    return (bias_weight, mae_weight, rmse_weight)


# def main():
if __name__ == '__main__':

    sample_folder = 'v4_conus_ic_pct_2010_2020'
    sample_block_size = 9

    output_filename_prefix = 'conus_isp_post_processing_binary_is_ndvi015_sm'

    (array_reference_isp_sum,
     array_annual_nlcd_isp_sum,
     array_conus_isp_sum) = prepare_evaluation_ard_data(sample_folder=sample_folder,
                                                        sample_block_size=sample_block_size,
                                                        array_resolution=np.array([30, 90, 150, 210, 270]),
                                                        output_filename_prefix=output_filename_prefix)

    # get the weight of different stratum
    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    array_target_year = np.arange(2010, 2021)

    (weight_ns_mean, weight_is_mean) = get_ns_is_weight(data_flag=data_flag,
                                                        isp_folder=isp_folder,
                                                        array_target_year=array_target_year)

    i_resolution = 0  # 30 m

    array_reference_isp = array_reference_isp_sum[i_resolution, :]
    array_annual_nlcd_isp = array_annual_nlcd_isp_sum[i_resolution, :]
    array_conus_isp = array_conus_isp_sum[i_resolution, :]

    print('Calculating unbiased accuracy of generated CONUS %ISA dataset...')
    (bias_weight, mae_weight, rmse_weight) = calculate_unbias_accuracy(array_conus_isp=array_conus_isp,
                                                                       array_reference_isp=array_reference_isp,
                                                                       weight_ns_mean=weight_ns_mean,
                                                                       weight_is_mean=weight_is_mean,
                                                                       print_flag=False)

    print(bias_weight, mae_weight, rmse_weight)


    print('Calculating unbiased accuracy of generated Annual NLCD %ISA dataset...')
    (bias_weight, mae_weight, rmse_weight) = calculate_unbias_accuracy(array_conus_isp=array_annual_nlcd_isp,
                                                                       array_reference_isp=array_reference_isp,
                                                                       weight_ns_mean=weight_ns_mean,
                                                                       weight_is_mean=weight_is_mean,
                                                                       print_flag=False)

    print(bias_weight, mae_weight, rmse_weight)


















