"""
    adjust the IS pct area using the regression analysis based on all sample points

    All sample points were not strictly random, but it can provide smaller variation
"""

import numpy as np
import os
from os.path import join, exists
import sys
import glob
import pandas as pd
from osgeo import gdal, gdal_array, gdalconst
import seaborn as sns
import matplotlib
import statsmodels.api as sm
from scipy import stats
import geopandas as gpd

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Error_statistical import Error_statistical

from sample_based_analysis.conus_is_pct_accuracy_unbias import (get_ns_is_weight,)
from conus_isp_analysis.conus_is_change_stats_plot import (get_conus_is_area_pct)

from analysis.utils_isp_change_stats_analysis import (sum_plot_is_change)

from sample_based_analysis.conus_is_pct_accuracy import (get_evaluation_sample_chips)

from uncertainty_estimation.utils_conus_is_area_uncertainty_est import (get_conus_is_change_summary_data,
                                                                        read_sample_based_conus_is_pct,
                                                                        plot_adjusted_are_with_uncertainty)

from uncertainty_estimation.is_area_adjustment_reg import (cal_isp_reg_confidence_interval,
                                                           plot_reg_confidence_interval,
                                                           adjust_conus_is_area_each_year_based_on_reg)


def get_weight_regression_results(data_flag, isp_folder):
    """

    :return:
    """

    # get the sample-based IS pct from 2010 to 2020
    array_target_year = np.arange(2010, 2021)

    sample_folder = 'v4_conus_ic_pct_2010_2020'
    sample_block_size = 9

    path_sample_folder = join(rootpath, 'results', 'accuracy_assessment', 'conus_is_pct', sample_folder)

    output_filename_gpkg = join(path_sample_folder, f'{sample_folder}_is_pct_sample.gpkg')
    gpd_sample = gpd.read_file(output_filename_gpkg, layer=f'{sample_folder}_sample')

    output_filename_prefix = 'conus_isp_post_processing_binary_is_ndvi015_sm'

    (img_reference_isp,
     img_annual_nlcd_isp,
     img_conus_isp) = get_evaluation_sample_chips(gpd_sample=gpd_sample,
                                                  sample_block_size=sample_block_size,
                                                  path_sample_folder=path_sample_folder,
                                                  sample_folder=sample_folder,
                                                  output_filename_prefix=output_filename_prefix)

    array_sample_conus_isp = img_conus_isp.flatten()
    array_sample_reference_isp = img_reference_isp.flatten()

    (weight_ns_mean_conus, weight_is_mean_conus) = get_ns_is_weight(data_flag=data_flag,
                                                                    isp_folder=isp_folder,
                                                                    array_target_year=array_target_year)

    # weighted regression between the sample and CONUS IS percentage data
    x = array_sample_conus_isp.reshape(-1, 1)
    y = array_sample_reference_isp.reshape(-1, 1)

    w = np.where(x == 0,
                 weight_ns_mean_conus,
                 weight_is_mean_conus)  # get the weight for each sample point

    w = w / np.nansum(w)  # normalize the weight

    # use statsmodels to get the regression summary, the results is the same as sklearn but can also estimate the 95% CI
    X = sm.add_constant(x)  # adding a constant to get the incercept
    model = sm.WLS(y, X, weights=w)
    results = model.fit()

    return (x, y, results)


# def main():
if __name__ =='__main__':

    data_flag = 'conus_isp'   # 'conus_isp' or 'annual_nlcd'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    title = 'Binary IS & NDVI015_SM'
    title_1 = f'{title}: IS area change'
    title_2 = f'{title}: year-to-year IS area change'

    (x, y, results) = get_weight_regression_results(data_flag=data_flag,
                                            isp_folder=isp_folder)

    # print(results.summary())

    (x_reg_plot,
     y_reg,
     ci_lower,
     ci_upper) = cal_isp_reg_confidence_interval(results,
                                                 x_input=np.arange(0, 101))

    intercept, slope = results.params
    print(f'intercept: {intercept}, slope: {slope}')

    # plot the scatter plot with weighted regression line
    plot_reg_confidence_interval(x, y,
                                 x_reg_plot,
                                 y_reg,
                                 ci_upper,
                                 ci_lower,
                                 flag_1_to_1=True,
                                 xlim=(-2, 120),
                                 ylim=(-2, 120),
                                 figsize=(12, 10),
                                 x_label='CONUS ISP',
                                 y_label='Reference',
                                 gridsize=50,
                                 bins='log',
                                 cbar_label='Count',
                                 title=None,
                                 )

    ##
    # map-based IS pct
    df_conus_is_area_conus = get_conus_is_area_pct(data_flag='conus_isp', isp_folder=isp_folder)

    array_target_year = np.arange(2010, 2021)

    array_map_conus_isp = df_conus_is_area_conus['area_pct'].values[np.isin(df_conus_is_area_conus['year'].values,
                                                                            array_target_year)]
    print(f'Map-based ISP using our CONUS ISP: {np.nanmean(array_map_conus_isp)}')

    # calculate the uncertainty range using the 95% CI
    (array_map_conus_isp,
     array_adjust_conus_isp,
     ci_upper_adjust,
     ci_lower_adjust) = cal_isp_reg_confidence_interval(results,
                                                        x_input=array_map_conus_isp)

    print(f'Adjusted Map-based ISP using our CONUS ISP: {np.nanmean(array_adjust_conus_isp)}')
    print(f'95% CI of adjusted CONUS ISP: {ci_lower_adjust[0]:.4f} to {ci_upper_adjust[0]:.4f}')

    # read the map-based CONUS IS area change from 1988 to 2020
    (df_is_change_sum_conus_plot,
     img_sum_isp_change_stats_conus_plot,
     img_sum_isp_change_stats_diag_zero_conus_plot,) = get_conus_is_change_summary_data(output_folder=join(rootpath, 'results', 'isp_change_stats',
                                                                                                           'conus_summary', data_flag, isp_folder),
                                                                                        array_year_plot=np.arange(1988, 2021, 1),)

    print(f'Original CONUS IS area change from 1988 to 2020: ')
    is_area_1988_2020 = np.concatenate([df_is_change_sum_conus_plot['is_area_year_1'].values,
                                       df_is_change_sum_conus_plot['is_area_year_2'].values[-1:]])
    is_area_1988_2020 = is_area_1988_2020 / 1e6  # convert to km2

    total_area = df_is_change_sum_conus_plot['total_area'].values[0] / 1e6  # convert to km2
    is_pct = is_area_1988_2020 / total_area * 100

    print(f'1988: {is_area_1988_2020[0]:.0f} km2 {is_pct[0]:.2f}% of total area')
    print(f'2020: {is_area_1988_2020[-1]:.0f} km2 {is_pct[-1]:.2f}% of total area')
    print(f'Change: {is_area_1988_2020[-1] - is_area_1988_2020[0]:.0f} km2  {is_pct[-1] - is_pct[0]:.2f}% of total area')


    # adjust the IS area in each year based on the regression results
    df_is_change_sum_conus_plot_adjusted = adjust_conus_is_area_each_year_based_on_reg(df_is_change_sum_conus_plot,
                                                                                       results=results)

    plot_adjusted_are_with_uncertainty(df_is_change_sum=df_is_change_sum_conus_plot_adjusted,
                                       title=None,
                                       x_label='Year',
                                       y_label='Area (km^2)',
                                       y_label_right='Area percentage (%)',
                                       x_axis_interval=2,
                                       y_axis_interval=None,
                                       right_decimals=3,
                                       figsize=(18, 10),
                                       xlim=None,
                                       ylim=None,
                                       legend_flag=False,
                                       )

    ##
    is_area_adjusted_1988_2020 = np.concatenate([df_is_change_sum_conus_plot_adjusted['is_area_year_1_adjust'].values,
                                                 df_is_change_sum_conus_plot_adjusted['is_area_year_2_adjust'].values[-1:]])
    is_area_adjusted_1988_2020 = is_area_adjusted_1988_2020 / 1e6  # convert to km2

    total_area = df_is_change_sum_conus_plot_adjusted['total_area'].values[0] / 1e6  # convert to km2

    is_pct = is_area_adjusted_1988_2020 / total_area * 100

    print(f'Adjusted CONUS IS area change from 1988 to 2020: ')
    print(f'1988: {is_area_adjusted_1988_2020[0]:.0f} km2 {is_pct[0]:.2f}% of total area')
    print(f'2020: {is_area_adjusted_1988_2020[-1]:.0f} km2' f' {is_pct[-1]:.2f}% of total area')
    print(f'Change: {is_area_adjusted_1988_2020[-1] - is_area_adjusted_1988_2020[0]:.0f} km2  {is_pct[-1] - is_pct[0]:.2f} % of total area')

    is_area_adjusted_upper_ci = np.concatenate([df_is_change_sum_conus_plot_adjusted['is_area_year_1_ci_upper'].values,
                                             df_is_change_sum_conus_plot_adjusted['is_area_year_2_ci_upper'].values[-1:]])
    is_area_adjusted_upper_ci = is_area_adjusted_upper_ci / 1e6  # convert to km2
    is_pct_upper_ci = is_area_adjusted_upper_ci / total_area * 100

    is_area_adjusted_lower_ci = np.concatenate([df_is_change_sum_conus_plot_adjusted['is_area_year_1_ci_lower'].values,
                                                df_is_change_sum_conus_plot_adjusted['is_area_year_2_ci_lower'].values[-1:]])
    is_area_adjusted_lower_ci = is_area_adjusted_lower_ci / 1e6
    is_pct_lower_ci = is_area_adjusted_lower_ci / total_area * 100

    # print(f'1988 95% CI: {is_area_adjusted_lower_ci[0]:.0f} to {is_area_adjusted_upper_ci[0]:.0f}, {is_pct_lower_ci[0]:.2f} to {is_pct_upper_ci[0]:.2f}%')
    # print(f'2020 95% CI: {is_area_adjusted_lower_ci[-1]:.0f} km2 to {is_area_adjusted_upper_ci[-1]:.0f} km2, {is_pct_lower_ci[-1]:.2f} to {is_pct_upper_ci[-1]:.2f}%')
    # print(f'Change 95% CI: {is_area_adjusted_lower_ci[-1] - is_area_adjusted_lower_ci[0]:.0f} km2 to {is_area_adjusted_upper_ci[-1] - is_area_adjusted_upper_ci[0]:.0f} km2')

    print(f'1988 95% CI: ±{is_area_adjusted_upper_ci[0]-is_area_adjusted_lower_ci[0]:.0f} km2 ±{is_pct_upper_ci[0] - is_pct_lower_ci[0]:.4f}%')
    print(f'2020 95% CI: ±{is_area_adjusted_upper_ci[-1]-is_area_adjusted_lower_ci[-1]:.0f} km2 ±{is_pct_upper_ci[-1] - is_pct_lower_ci[-1]:.4f}%')
    print(f'Change 95% CI: {is_area_adjusted_lower_ci[-1] - is_area_adjusted_lower_ci[0]:.0f} km2 to {is_area_adjusted_upper_ci[-1] - is_area_adjusted_upper_ci[0]:.0f} km2')

    # calculate the ISA change from 1997 to 2020
    is_area_adjusted_1997_2020 = is_area_adjusted_1988_2020[9:]  # from 1997 to 2020
    print(f'Percent change from 1997 to 2020: ', f'{(is_area_adjusted_1997_2020[-1] - is_area_adjusted_1997_2020[0]) / is_area_adjusted_1997_2020[0] * 100:.2f}%')








