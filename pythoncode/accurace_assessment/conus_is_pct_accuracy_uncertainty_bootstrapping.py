"""
    calculate the uncertainty of CONUS IS percentage accuracy by using bootstrapping strategy

    We have 30 sample blocks for "ISP = 0" stratum, and 20 sample blocks for "ISP > 0" stratum.

    The code implements two bootstrapping strategies:

    Bootstrapping strategy #1 is used as the final strategy
"""

import numpy as np
import pandas as pd
import os
from os.path import join, exists
import sys
from osgeo import ogr, gdal, osr, gdalconst, gdal_array
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = os.path.join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


from pythoncode.accurace_assessment.conus_is_pct_accuracy_unbias import (get_ns_is_weight,
                                                                         prepare_evaluation_ard_data,
                                                                         calculate_unbias_accuracy)


def bootstrapping_strategy_one(bootstrapping_times, bootstrapping_size,
                               array_reference_isp, array_conus_isp,
                               weight_ns_mean, weight_is_mean):
    """
        Implementation of the first bootstrapping strategy

        (1) Randomly select 10/20/30 sample blocks from the 50 sample pool
        (2) Based on the weight, calculate the unbiased accuracy (Bias, MAE, RMSE) for the selected sample blocks
        (3) Repeat the above steps 1000 times
        (4) Get the 95% confidence interval for the accuracy (2.5th and 97.5th percentiles)

        The uncertainty can be affected by the bootstrapping size (10/20/30)

        :param bootstrapping_times:
        :param bootstrapping_size: number of sample blocks in each bootstrapping round
        :param array_reference_isp:
        :param array_conus_isp:
        :param weight_ns_mean:
        :param weight_is_mean:
        :return:
    """

    df_bootstrapping_accuracy = pd.DataFrame(columns=['bootstrapping_id', 'index_select',
                                                      'bias_weight', 'mae_weight', 'rmse_weight'],
                                             index=np.arange(0, bootstrapping_times))

    for i_bootstrapping in range(0, bootstrapping_times):
        index_select = np.random.choice(np.arange(0, len(array_reference_isp)), size=bootstrapping_size, replace=False)
        print(i_bootstrapping, index_select)

        array_reference_isp_select = array_reference_isp[index_select]
        # array_annual_nlcd_isp_select = array_annual_nlcd_isp[index_select]
        array_conus_isp_select = array_conus_isp[index_select]

        (bias_weight, mae_weight, rmse_weight) = calculate_unbias_accuracy(array_conus_isp_select,
                                                                           array_reference_isp_select,
                                                                           weight_ns_mean,
                                                                           weight_is_mean,
                                                                           print_flag=False)

        # print(f'bias_weight: {bias_weight}, mae_weight: {mae_weight}, rmse_weight: {rmse_weight}')

        df_bootstrapping_accuracy.loc[i_bootstrapping, 'bootstrapping_id'] = i_bootstrapping + 1
        df_bootstrapping_accuracy.loc[i_bootstrapping, 'index_select'] = index_select

        df_bootstrapping_accuracy.loc[i_bootstrapping, 'bias_weight'] = bias_weight
        df_bootstrapping_accuracy.loc[i_bootstrapping, 'mae_weight'] = mae_weight
        df_bootstrapping_accuracy.loc[i_bootstrapping, 'rmse_weight'] = rmse_weight

    return df_bootstrapping_accuracy


def hist_plot_stats(data_plot,
                    title=None,
                    x_label='',
                    y_label='',
                    bins='auto',
                    binrange=None,
                    kde=False,
                    stats='count',
                    figsize=(14, 8),
                    discrete=False,
                    xlim=None,
                    x_ticks_rotation=0,
                    tick_label_size=18,
                    sns_style="whitegrid",
                    color='#75bbfd'):
    """
        histogram plot the stats of the data
    """

    sns.set_style(sns_style)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    legend_size = 20
    tick_label_size = tick_label_size
    axis_label_size = 24
    title_label_size = 28
    tick_length = 4

    ax.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    ax.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

    ax.set_xlabel(x_label, size=axis_label_size)
    ax.set_ylabel(y_label, size=axis_label_size)

    ax = sns.histplot(data=data_plot,
                      stat=stats,
                      bins=bins,
                      kde=kde,
                      binrange=binrange,
                      discrete=discrete,
                      alpha=1.0,
                      color=color,
                      )

    ax.set_title(title, size=title_label_size)
    plt.xlim(xlim)
    # ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))

    plt.xticks(rotation=x_ticks_rotation)
    # plt.legend(loc='best', fontsize=legend_size)

    plt.tight_layout()
    plt.show()


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

    # get the evaluation sample data at 30-meter resolution
    array_reference_isp = array_reference_isp_sum[0, :]
    array_annual_nlcd_isp = array_annual_nlcd_isp_sum[0, :]
    array_conus_isp = array_conus_isp_sum[0, :]

    # get the weight of different stratum
    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    array_target_year = np.arange(2010, 2021)

    (weight_ns_mean, weight_is_mean) = get_ns_is_weight(data_flag=data_flag,
                                                        isp_folder=isp_folder,
                                                        array_target_year=array_target_year)

    ##
    (bias_weight, mae_weight, rmse_weight) = calculate_unbias_accuracy(array_conus_isp,
                                                                       array_reference_isp,
                                                                       weight_ns_mean,
                                                                       weight_is_mean,
                                                                       print_flag=True)

    print(f'bias_weight: {bias_weight}, mae_weight: {mae_weight}, rmse_weight: {rmse_weight}')

    ## bootstrapping strategy #1
    bootstrapping_times = 100
    # bootstrapping_size = 20   # the number of sample blocks for each bootstrapping

    for bootstrapping_size in [20]:

        df_bootstrapping_accuracy_one = bootstrapping_strategy_one(bootstrapping_times=bootstrapping_times,
                                                                   bootstrapping_size=bootstrapping_size,
                                                                   array_reference_isp=array_reference_isp,
                                                                   array_conus_isp=array_conus_isp,
                                                                   weight_ns_mean=weight_ns_mean,
                                                                   weight_is_mean=weight_is_mean)

        # get the 95% confidence interval for the accuracy (2.5th and 97.5th percentiles)
        bias_weight_95 = np.percentile(df_bootstrapping_accuracy_one['bias_weight'], [2.5, 97.5])
        mae_weight_95 = np.percentile(df_bootstrapping_accuracy_one['mae_weight'], [2.5, 97.5])
        rmse_weight_95 = np.percentile(df_bootstrapping_accuracy_one['rmse_weight'], [2.5, 97.5])

        print(f'bias_weight_95: {bias_weight_95}, mae_weight_95: {mae_weight_95}, rmse_weight_95: {rmse_weight_95}')

        bias_mean_bootstrap = np.nanmean(df_bootstrapping_accuracy_one['bias_weight'])
        mae_mean_bootstrap = np.nanmean(df_bootstrapping_accuracy_one['mae_weight'])
        rmse_mean_bootstrap = np.nanmean(df_bootstrapping_accuracy_one['rmse_weight'])

        print(f'bias_weight mean: {bias_mean_bootstrap}, mae_weight mean: {mae_mean_bootstrap}, rmse_weight mean: {rmse_mean_bootstrap}')

        hist_plot_stats(df_bootstrapping_accuracy_one['bias_weight'].values,
                        title=f'{bootstrapping_times} Bootstrapping bias {bootstrapping_size} 95% CI [{bias_weight_95[0]:.2f} {bias_weight_95[1]:.2f}]',
                        x_label='Bias', )

        hist_plot_stats(df_bootstrapping_accuracy_one['mae_weight'].values,
                        title=f'{bootstrapping_times} Bootstrapping MAE {bootstrapping_size} 95% CI [{mae_weight_95[0]:.2f} {mae_weight_95[1]:.2f}]',
                        x_label='MAE', )

        hist_plot_stats(df_bootstrapping_accuracy_one['rmse_weight'].values,
                        title=f'{bootstrapping_times} Bootstrapping RMSE {bootstrapping_size} 95% CI [{rmse_weight_95[0]:.2f} {rmse_weight_95[1]:.2f}]',
                        x_label='RMSE', )

        # output_folder = join(rootpath, 'results', 'accuracy_assessment', 'conus_is_pct', sample_folder,
        #                      f'conus_isp_bootstrapping_results', )
        # if not exists(output_folder):
        #     os.makedirs(output_folder, exist_ok=True)
        #
        # output_filename = join(output_folder,
        #                        f'conus_isp_{output_filename_prefix}_bootstrapping_{bootstrapping_size}_accuracy_{bootstrapping_times}.csv')
        #
        # df_bootstrapping_accuracy_one.to_csv(output_filename, index=False)













