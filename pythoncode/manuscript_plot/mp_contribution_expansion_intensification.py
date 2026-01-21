"""
    code to plot the CONUS IS increase contribution from expansion and intensification for the manuscript

    The proportion of IS expansion to IS increase is around 82.86%
    The proportion of IS intensification to IS increase is around 17.14%

    From 1988 to 2020, the proportion of IS expansion to IS increase is decreasing over time, from around 85.73% in 1988 to around 75.72% in 2020
    The proportion of IS intensification to IS increase is increasing over time, from around 14.27% in 1988 to around 24.28% in 2020
"""

import numpy as np
import os
from os.path import join, exists
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import seaborn as sns
import matplotlib
import matplotlib.ticker as plticker
import matplotlib.ticker as ticker
from scipy import stats
import pymannkendall as mk

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


# def main():
if __name__ =='__main__':

    ##
    data_flag = 'conus_isp'   # 'conus_isp' or 'annual_nlcd'

    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    output_folder = (join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag, isp_folder)
        if data_flag == 'conus_isp' else join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag))

    print(data_flag, isp_folder)

    df_is_change_sum_conus = pd.read_csv(join(output_folder, 'conus_is_change_type_summary.csv'))

    array_year_plot = np.arange(1988, 2021, 1)

    # define the time period for the plot
    df_is_change_sum_conus_plot = df_is_change_sum_conus[np.isin(df_is_change_sum_conus['year_1'].values, (array_year_plot[0:-1]))].copy()

    list_year = df_is_change_sum_conus_plot['year_2'].values

    ##
    # proportion of IS expansion and intensification contributing to IS increase is increasing over time
    prop_is_expansion_annual = df_is_change_sum_conus_plot['area_is_expansion'] / (df_is_change_sum_conus_plot['area_is_expansion'] + df_is_change_sum_conus_plot['area_is_intensification']) * 100
    prop_is_intensification_annual = df_is_change_sum_conus_plot['area_is_intensification'] / (df_is_change_sum_conus_plot['area_is_expansion'] + df_is_change_sum_conus_plot['area_is_intensification']) * 100

    prop_is_expansion_annual = prop_is_expansion_annual.values
    prop_is_intensification_annual = prop_is_intensification_annual.values

    # df_contribution = pd.DataFrame({
    #     'year': df_is_change_sum_conus_plot['year_2'],
    #     'prop_is_expansion_annual': prop_is_expansion_annual,
    #     'prop_is_intensification_annual': prop_is_intensification_annual})

    # mk_results_expansion = mk.original_test(prop_is_expansion_annual)
    # mk_results_intensification = mk.original_test(prop_is_intensification_annual)

    array_theilslopes = np.arange(0, len(prop_is_expansion_annual), 1)
    array_theilslopes = array_theilslopes - np.nanmean(array_theilslopes)   # force the line to always pass through the mean of x

    (slope_expansion,
     intercept_expansion,
     lo_ci_expansion,
     hi_ci_expansion) = stats.theilslopes(prop_is_expansion_annual, array_theilslopes, alpha=0.95, method='joint')

    (slope_intensification,
     intercept_intensification,
     lo_ci_intensification,
     hi_ci_intensification) = stats.theilslopes(prop_is_intensification_annual, array_theilslopes, alpha=0.95, method='joint')

    array_expansion_hi_ci = intercept_expansion + hi_ci_expansion * array_theilslopes
    array_expansion_lo_ci = intercept_expansion + lo_ci_expansion * array_theilslopes

    array_expansion_ori_reg = intercept_expansion + slope_expansion * array_theilslopes

    array_intensification_hi_ci = intercept_intensification + hi_ci_intensification * array_theilslopes
    array_intensification_lo_ci = intercept_intensification + lo_ci_intensification * array_theilslopes

    array_intensification_ori_reg = intercept_intensification + slope_intensification * array_theilslopes

    ##
    # plot the relative proportion of IS expansion and intensification in contributing the IS increase
    flag_plot_vline = True

    plt.rcParams['font.family'] = 'Arial'

    fig, ax_plot = plt.subplots(ncols=1, nrows=1, figsize=(14, 8))

    tick_label_size = 24
    title_size = 22
    tick_length = 8

    line_width = 3.0
    linestyle = 'solid'

    for spine in ax_plot.spines.values():
        spine.set_linewidth(3.0)

    ax_plot.scatter(list_year, prop_is_expansion_annual,
                    label='IS expansion', color='#ff0000',
                    marker='o', s=135,)

    # ax_plot.plot(list_year, array_expansion_ori_reg, color='#ff0000',
    #              linestyle=linestyle, linewidth=line_width)
    #
    # ax_plot.fill_between(list_year, array_expansion_hi_ci, array_expansion_lo_ci,
    #                      color='#ff0000', alpha=0.2)

    ax_plot.scatter(list_year, prop_is_intensification_annual,
             label='IS intensification', color='#7e1e9c',
             marker='^', s=160,
             # linestyle=linestyle, linewidth=line_width
             )

    # ax_plot.plot(list_year, array_intensification_ori_reg, color='#7e1e9c',
    #              linestyle=linestyle, linewidth=line_width)

    # ax_plot.plot(list_year, array_intensification_hi_ci, color='#7e1e9c', linestyle='dashed', linewidth=2.5)
    # ax_plot.plot(list_year, array_intensification_lo_ci, color='#7e1e9c', linestyle='dashed', linewidth=2.5)

    # ax_plot.fill_between(list_year, array_intensification_hi_ci, array_intensification_lo_ci,
    #                      color='#7e1e9c', alpha=0.2)

    ax_plot.tick_params('x', labelsize=tick_label_size, direction='out',
                        length=tick_length, bottom=True, which='major',
                        width=2.5)
    ax_plot.tick_params('y', labelsize=tick_label_size, direction='out',
                        length=tick_length, left=True, which='major',
                        width=2.5)
    ax_plot.set_yticks(np.arange(10, 91, 10))

    # ax_plot.set_ylim(50, 92)
    # ax_plot.set_yticks(np.array([75, 80, 85]))

    ax_plot.xaxis.set_major_locator(plticker.MultipleLocator(base=2.0))
    # Create a FuncFormatter to change the four-digit year to two-digit year
    ax_plot.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x) % 100:02d}'))

    if flag_plot_vline:
        assert 2008 in list_year, '2008 is not in the GDP year list'
        ax_plot.axvline(x=list_year[list_year == 2008][0],
                               color='#fa7305',
                               linestyle='dashed',
                               linewidth=3.0)

    # ax2 = ax_plot.twinx()
    # ax2.plot(list_year, prop_is_intensification_annual,
    #          label='IS intensification', color='#7e1e9c',
    #          marker='^', markersize=16,
    #          linestyle=linestyle, linewidth=line_width
    #          )
    #
    # ax2.set_ylim(12, 40)
    # ax2.set_yticks(np.array([15, 20, 25]))

    # ax2.tick_params('x', labelsize=tick_label_size, direction='out',
    #                 length=tick_length, bottom=True, which='major', width=2.5)
    # ax2.tick_params('y', labelsize=tick_label_size, direction='out',
    #                 length=tick_length, left=False, which='major', width=2.5,
    #                 colors='#7e1e9c')

    plt.tight_layout()
    plt.show()

    # plt.savefig(join(r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_expansion_intensification',
    #                  'relative_contribution_IS_expansion_intensification_conus.jpg'),
    #             dpi=600)









