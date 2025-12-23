"""
    plot the CONUS impervious surface area change for the manuscript
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

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from evaluation.utils_evaluation import convert_8_tile_names_to_6_tile_names, convert_6_tile_names_to_8_tile_names
# from analysis.utils_isp_change_stats_analysis import (plot_is_pct_change_between_two_years,
#                                                       plot_isp_change_type_ts,
#                                                       plot_is_area_ts,
#                                                       sum_plot_is_change)

from manuscript_plot.utils_manuscript_is_area_plot import (manuscript_is_area_plot)
from conus_isp_financial_crisis.conus_is_fc_impact_recovery import (print_is_reduction_recovery)

from uncertainty_estimation.is_change_adjustment_reg import (get_weight_regression_results,
                                                                 get_conus_is_change_summary_data,
                                                                 generate_adjusted_conus_is_change_dataframe)

from uncertainty_estimation.print_adjust_is_area import (print_adjust_is_area_change_stats)



# def main():
if __name__ =='__main__':

    ##
    data_flag = 'conus_isp'   # 'conus_isp' or 'annual_nlcd'

    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    # title = 'Binary IS & NDVI015_SM'
    title = 'CONUS ISP'

    # isp_folder = 'individual_year_tile_post_processing_is_expansion_ndvi015_sm'
    # title = 'IS expansion & NDVI015'

    if data_flag == 'conus_isp':
        # from 1985 to 2021, 2022 is not included, because the last change stats is from 2021 to 2022
        # array_isp_change_stats_year = np.arange(1985, 2022, 1)
        output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag, isp_folder)

        title_1 = f'{title}: IS area change'
        title_2 = f'{title}: year-to-year IS area change'

    elif data_flag == 'annual_nlcd':

        # array_isp_change_stats_year = np.arange(1985, 2023, 1)
        output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag)

        title_1 = f'Annual NLCD ISP: IS area change'
        title_2 = f'Annual NLCD ISP: year-to-year IS area change'
    else:
        raise ValueError('data_flag is not recognized')

    print(data_flag, isp_folder, title)

    ##
    # define the time period for the plot
    # array_year_plot = np.arange(1980, 2025, 1)
    array_year_plot = np.arange(1988, 2021, 1)

    (x, y, results) = get_weight_regression_results(data_flag=data_flag,
                                                    isp_folder=isp_folder)

    (df_is_change_sum_conus_plot,
     img_sum_isp_change_stats_conus_plot,
     img_sum_isp_change_stats_diag_zero_conus_plot,) = get_conus_is_change_summary_data(output_folder=output_folder,
                                                                                        array_year_plot=array_year_plot)

    df_is_change_sum_conus_plot_adjust = generate_adjusted_conus_is_change_dataframe(df_is_change_sum_conus_plot,
                                                                                     results,
                                                                                     img_sum_isp_change_stats_diag_zero_conus_plot)
    ##
    print('CONUS IS area change stats summary:')
    print_is_reduction_recovery(df_is_change_sum_conus_plot)

    print()

    print('Adjusted CONUS IS area change stats summary:')
    # print_is_reduction_recovery_adjust(df_is_change_sum_conus_plot_adjusted)
    print_adjust_is_area_change_stats(df_is_change_sum_conus_plot_adjust=df_is_change_sum_conus_plot_adjust)
    ##
    # plot the IS change

    title_1 = None
    title_2 = None

    figsize = (25, 7)
    xlim = None
    ylim_1 = None
    ylim_2 = None
    output_flag = False

    if data_flag == 'conus_isp':
        output_filename = r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend\CONUS_IS_area_change.jpg'
    elif data_flag == 'annual_nlcd':
        output_filename = r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend\Annual_NLCD_IS_area_change.jpg'
    sns_style = 'white'
    legend_flag = False
    plot_flag = 'area'
    flag_highlight_2008 = False

    manuscript_is_area_plot(df_is_change_sum_conus_plot,
                            title_1, title_2,
                            figsize=figsize,
                            xlim=xlim, ylim_1=ylim_1, ylim_2=ylim_2,
                            output_flag=output_flag,
                            output_filename=output_filename,
                            sns_style=sns_style,
                            legend_flag=legend_flag,
                            plot_flag=plot_flag,
                            flag_highlight_2008=flag_highlight_2008)

    ##
    # adjust the IS area and plot with uncertainty interval

    output_filename_adjust = r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend\CONUS_IS_area_change_adjust.jpg'

    manuscript_is_area_plot(df_is_change_sum_conus_plot_adjust,
                            title_1,
                            title_2,
                            figsize=figsize,
                            xlim=xlim, ylim_1=ylim_1, ylim_2=ylim_2,
                            output_flag=False,
                            output_filename=output_filename_adjust,
                            sns_style=sns_style,
                            legend_flag=legend_flag,
                            plot_flag=plot_flag,
                            flag_highlight_2008=flag_highlight_2008,
                            flag_adjust_with_ci=True,
                            fill_alpha=0.3)












