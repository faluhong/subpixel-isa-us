"""
    adjust the IS pct area using the regression analysis based on all sample points

    All sample points were not strictly random, but it can provide smaller variation
"""

import numpy as np
import os
from os.path import join
import sys
import statsmodels.api as sm
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import stats

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


from pythoncode.accurace_assessment.conus_is_pct_accuracy_unbias import (get_ns_is_weight)
from pythoncode.accurace_assessment.conus_is_pct_accuracy import (get_evaluation_sample_chips)

from pythoncode.accurace_assessment.utils_conus_is_area_uncertainty_est import (get_conus_is_change_summary_data,
                                                                                plot_adjusted_are_with_uncertainty)


def get_weight_regression_results(data_flag, isp_folder):
    """

    :return:
    """

    # get the sample-based IS pct from 2010 to 2020
    array_target_year = np.arange(2010, 2021)

    sample_folder = 'v4_conus_ic_pct_2010_2020'
    sample_block_size = 9

    path_sample_folder = join(rootpath, 'results', 'accuracy_assessment', 'conus_is_pct')

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


def get_conus_is_area_pct(data_flag, isp_folder):
    """
        get the dataframe recording the annual CONUS IS area and percentage

        :param data_flag: 'conus_isp' or 'annual_nlcd'
        :param isp_folder: the folder name of the ISP data, e.g., 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
                           coule be None if data_flag is 'annual_nlcd'
        :return:
    """

    if data_flag == 'conus_isp':
        # from 1985 to 2021, 2022 is not included, because the last change stats is from 2021 to 2022
        output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag, isp_folder)
    elif data_flag == 'annual_nlcd':
        output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag)
    else:
        raise ValueError('data_flag is not recognized')

    df_is_change_sum_conus = pd.read_csv(join(output_folder, 'conus_is_change_type_summary.csv'))

    array_year_plot = np.concatenate([df_is_change_sum_conus['year_1'].values, np.array([df_is_change_sum_conus['year_2'].values[-1]])])

    array_is_area = np.concatenate([df_is_change_sum_conus['is_area_year_1'].values, np.array([df_is_change_sum_conus['is_area_year_2'].values[-1]])])
    array_is_area = array_is_area / 1000000  # convert the area to km^2

    array_is_pct = array_is_area / (df_is_change_sum_conus['total_area'].values[0] / 1000000) * 100

    df_conus_isp = pd.DataFrame({'year': array_year_plot,
                                 'area_km2': array_is_area,
                                 'area_pct': array_is_pct})

    return df_conus_isp


def cal_isp_reg_confidence_interval(weight_reg_summary,
                                    x_input=np.arange(0, 101)):
    """
        get the 95% confidence interval of the weighted regression line

        :param weight_reg_summary:
        :return:
            x_reg_plot: x values for regression line plot, default is np.arange(0, 101), ISP from 0 to 100
            ci_upper: upper bound of 95% CI
            ci_lower: lower bound of 95% CI
    """

    X_pred = np.column_stack([np.ones_like(x_input), x_input])  # if you used add_constant()

    params = weight_reg_summary.params
    cov = weight_reg_summary.cov_params()

    # Predicted mean line
    y_pred = X_pred @ params

    # Standard error of predicted mean:
    se_pred = np.sqrt(np.sum(X_pred @ cov * X_pred, axis=1))

    # Get critical t value
    df = weight_reg_summary.df_resid
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Confidence interval for the prediction of the *mean*
    ci_upper = y_pred + t_crit * se_pred
    ci_lower = y_pred - t_crit * se_pred

    # if the input_x is a single value, return the value instead of array
    if np.isscalar(x_input):
        y_pred = y_pred[0]
        ci_upper = ci_upper[0]
        ci_lower = ci_lower[0]

    return (x_input, y_pred, ci_upper, ci_lower)


def plot_reg_confidence_interval(x, y,
                                 x_reg_plot,
                                 y_reg,
                                 ci_upper,
                                 ci_lower,
                                 flag_1_to_1=True,
                                 xlim=(-2, 140),
                                 ylim=(-2, 140),
                                 figsize=(12, 10),
                                 x_label='CONUS ISP',
                                 y_label='Reference',
                                 gridsize=50,
                                 bins=None,
                                 cbar_label='Count',
                                 title=None,
                                 ):
    """
        plot the scatter, regression line with 95% CI
    """

    matplotlib.rcParams['font.family'] = "Arial"

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    tick_label_size = 28
    axis_label_size = 30
    cbar_tick_label_size = 24
    title_label_size = 32
    tick_length = 4

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    # grids containing at least one point will be diplayed
    img = plt.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=1)

    # img = plt.scatter(x, y, marker='o', s=120,)

    if flag_1_to_1 == True:
        # ax.plot([0, 100], [0, 100], color='#363737', linestyle='--', linewidth=3)
        ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]],
                color='#363737', linestyle='--', linewidth=3)

    ax.plot(x_reg_plot, y_reg, color='red',
            linestyle='solid', linewidth=3,
            label='weighted regression line')

    # Confidence band
    plt.fill_between(x_reg_plot, ci_lower, ci_upper,
                     color='red',
                     alpha=0.4,
                     label="95% CI")

    # plot the original regression line and error statistics
    # errors_stats_plot = Error_statistical(x, y)
    # errors_stats_plot.print_error_stats()

    # y_reg = errors_stats_plot.slope * np.arange(0, 101) + errors_stats_plot.intercept
    # ax.plot(np.arange(0, 101), y_reg, color='black',
    #         linestyle='solid', linewidth=3,
    #         label='original regression line')

    # Adding colorbar
    cb = plt.colorbar(img, cmap=cmap)
    cb.ax.tick_params(labelsize=cbar_tick_label_size)
    cb.ax.set_ylabel(cbar_label, size=axis_label_size)

    ax.tick_params('x', labelsize=tick_label_size, direction='out',
                   length=tick_length, bottom=True)
    ax.tick_params('y', labelsize=tick_label_size, direction='out',
                   length=tick_length, left=True)

    ax.set_xlabel(x_label, size=axis_label_size)
    ax.set_ylabel(y_label, size=axis_label_size)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_title(title, size=title_label_size)

    ax.legend(fontsize=20, loc='best')

    plt.tight_layout()
    plt.show()


def adjust_conus_is_area_each_year_based_on_reg(df_is_change_sum_conus_plot, results):
    """
        adjust the IS area in each year based on the regression results

        define a new dataframe to store the adjusted IS area and 95% CI

        Note: 2025-12-09 The function is not recommended for usage. It may cause confusion since the original area columns are overwritten.
                         It's better to use generate_adjusted_conus_is_change_dataframe function

        :param df_is_change_sum_conus_plot:
        :param results:
        :return:
    """

    df_is_change_sum_conus_plot_adjusted = df_is_change_sum_conus_plot.copy()

    array_is_pct_year_1 = df_is_change_sum_conus_plot_adjusted['is_area_year_1'] / df_is_change_sum_conus_plot_adjusted['total_area'] * 100.0
    array_is_pct_year_2 = df_is_change_sum_conus_plot_adjusted['is_area_year_2'] / df_is_change_sum_conus_plot_adjusted['total_area'] * 100.0

    # array_is_pct_year_1_adjust = results.params[1] * array_is_pct_year_1 + results.params[0]
    # array_is_pct_year_2_adjust = results.params[1] * array_is_pct_year_2 + results.params[0]

    (array_is_pct_year_1,
     array_is_pct_year_1_adjust,
     ci_upper_is_pct_year_1,
     ci_lower_is_pct_year_1,) = cal_isp_reg_confidence_interval(results,
                                                                x_input=array_is_pct_year_1)

    (array_is_pct_year_2,
     array_is_pct_year_2_adjust,
     ci_upper_is_pct_year_2,
     ci_lower_is_pct_year_2,) = cal_isp_reg_confidence_interval(results,
                                                                x_input=array_is_pct_year_2)

    df_is_change_sum_conus_plot_adjusted['is_area_year_1_adjust'] = array_is_pct_year_1_adjust / 100.0 * df_is_change_sum_conus_plot_adjusted['total_area']
    df_is_change_sum_conus_plot_adjusted['is_area_year_2_adjust'] = array_is_pct_year_2_adjust / 100.0 * df_is_change_sum_conus_plot_adjusted['total_area']

    df_is_change_sum_conus_plot_adjusted['is_area_year_1_ci_lower'] = ci_lower_is_pct_year_1 / 100.0 * df_is_change_sum_conus_plot_adjusted['total_area']
    df_is_change_sum_conus_plot_adjusted['is_area_year_1_ci_upper'] = ci_upper_is_pct_year_1 / 100.0 * df_is_change_sum_conus_plot_adjusted['total_area']
    df_is_change_sum_conus_plot_adjusted['is_area_year_2_ci_lower'] = ci_lower_is_pct_year_2 / 100.0 * df_is_change_sum_conus_plot_adjusted['total_area']
    df_is_change_sum_conus_plot_adjusted['is_area_year_2_ci_upper'] = ci_upper_is_pct_year_2 / 100.0 * df_is_change_sum_conus_plot_adjusted['total_area']

    df_is_change_sum_conus_plot_adjusted['is_area_year_1'] = df_is_change_sum_conus_plot_adjusted['is_area_year_1_adjust']
    df_is_change_sum_conus_plot_adjusted['is_area_year_2'] = df_is_change_sum_conus_plot_adjusted['is_area_year_2_adjust']

    return df_is_change_sum_conus_plot_adjusted


# def main():
if __name__ =='__main__':

    data_flag = 'conus_isp'   # 'conus_isp' or 'annual_nlcd'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

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

    ## print the adjusted IS area change from 1988 to 2020
    is_area_adjusted_1988_2020 = np.concatenate([df_is_change_sum_conus_plot_adjusted['is_area_year_1_adjust'].values,
                                                 df_is_change_sum_conus_plot_adjusted['is_area_year_2_adjust'].values[-1:]])
    is_area_adjusted_1988_2020 = is_area_adjusted_1988_2020 / 1e6  # convert to km2

    total_area = df_is_change_sum_conus_plot_adjusted['total_area'].values[0] / 1e6  # convert to km2

    is_pct = is_area_adjusted_1988_2020 / total_area * 100

    print(f'Adjusted CONUS IS area change from 1988 to 2020: ')
    print(f'1988: {is_area_adjusted_1988_2020[0]:.0f} km2 {is_pct[0]:.2f}% of total area')
    print(f'2020: {is_area_adjusted_1988_2020[-1]:.0f} km2' f' {is_pct[-1]:.2f}% of total area')

    is_area_adjusted_upper_ci = np.concatenate([df_is_change_sum_conus_plot_adjusted['is_area_year_1_ci_upper'].values,
                                             df_is_change_sum_conus_plot_adjusted['is_area_year_2_ci_upper'].values[-1:]])
    is_area_adjusted_upper_ci = is_area_adjusted_upper_ci / 1e6  # convert to km2
    is_pct_upper_ci = is_area_adjusted_upper_ci / total_area * 100

    is_area_adjusted_lower_ci = np.concatenate([df_is_change_sum_conus_plot_adjusted['is_area_year_1_ci_lower'].values,
                                                df_is_change_sum_conus_plot_adjusted['is_area_year_2_ci_lower'].values[-1:]])
    is_area_adjusted_lower_ci = is_area_adjusted_lower_ci / 1e6
    is_pct_lower_ci = is_area_adjusted_lower_ci / total_area * 100

    print(f'1988 95% CI: ±{(is_area_adjusted_upper_ci[0] - is_area_adjusted_lower_ci[0]) / 2:.2f} km2 ±{(is_pct_upper_ci[0] - is_pct_lower_ci[0]) / 2:.4f}%')
    print(f'2020 95% CI: ±{(is_area_adjusted_upper_ci[-1]-is_area_adjusted_lower_ci[-1]) / 2:.2f} km2 ±{(is_pct_upper_ci[0] - is_pct_lower_ci[0]) / 2:.4f}%')














