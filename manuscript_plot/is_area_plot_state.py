"""
    plot the selected state IS area change for the manuscript

    Current selected states are: Texas and California
"""


import numpy as np
import os
from os.path import join, exists
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import matplotlib
import seaborn as sns
import matplotlib

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from evaluation.utils_evaluation import convert_8_tile_names_to_6_tile_names, convert_6_tile_names_to_8_tile_names
from analysis.utils_isp_change_stats_analysis import (generate_isp_change_summary_dataframe,)

from conus_isp_analysis.state_is_change_display import (get_isp_change_stats_output_file_single_state_all_year)
from manuscript_plot.utils_manuscript_is_area_plot import manuscript_is_area_plot

from uncertainty_estimation.is_area_adjustment_reg_all_sample import (get_weight_regression_results)

from uncertainty_estimation.is_change_adjustment_reg import (generate_adjusted_conus_is_change_dataframe)
from uncertainty_estimation.print_adjust_is_area import (print_adjust_is_area_change_stats)

# def main():
if __name__ == '__main__':

    filename_output = join(rootpath, 'data', 'shapefile', 'CONUS_boundary', f'tl_2023_us_state', 'conus_state_basic_info.csv')
    df_conus_state_basic_info = pd.read_csv(filename_output)

    # data_flag = 'annual_nlcd' # 'conus_isp' or 'annual_nlcd'
    rootpath_nlcd_directory = None
    nlcd_folder = 'NLCD_annual'
    nlcd_filter_ocean_flag = False

    data_flag = 'conus_isp'
    rootpath_conus_isp = None
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    if data_flag == 'conus_isp':
        array_target_year = np.arange(1985, 2022, 1)
        output_path_figure = join(rootpath, 'results', 'isp_change_stats', 'state_level', f'{data_flag}_figure', isp_folder)

    elif data_flag == 'annual_nlcd':
        array_target_year = np.arange(1985, 2023, 1)
        output_path_figure = join(rootpath, 'results', 'isp_change_stats', 'state_level', f'{data_flag}_figure')
    else:
        raise ValueError('The data_flag is not correct')

    ##
    # array_target_year = np.arange(1988, 2020, 1)

    # for i_state in range(0, len(df_conus_state_basic_info)):
    # for i_state in range(0, 1):
    for i_state in [13, 25]:  # Texas and California
        state_id = df_conus_state_basic_info['id'].values[i_state]
        state_short_name = df_conus_state_basic_info['STUSPS'].values[i_state]
        state_name = df_conus_state_basic_info['NAME'].values[i_state]

        print(state_name)

        (img_sum_isp_change_stats,
         img_sum_isp_change_stats_diag_zero) = get_isp_change_stats_output_file_single_state_all_year(data_flag,
                                                                                                      state_name,
                                                                                                      state_short_name,
                                                                                                      array_target_year,
                                                                                                      isp_folder=isp_folder,
                                                                                                      rootpath_conus_isp=None,
                                                                                                      rootpath_nlcd_directory=None)

        df_is_change_sum = generate_isp_change_summary_dataframe(img_sum_isp_change_stats, array_target_year)

        # array_year_plot = np.arange(1980, 2025, 1)
        array_year_plot = np.arange(1988, 2021, 1)

        # define the time period for the plot
        df_is_change_sum_plot = df_is_change_sum[np.isin(df_is_change_sum['year_1'].values, (array_year_plot[0:-1]))].copy()

        img_sum_isp_change_stats_plot = img_sum_isp_change_stats[np.isin(np.arange(1985, 2022, 1), (array_year_plot[0:-1])), :, :]
        img_sum_isp_change_stats_diag_zero_plot = img_sum_isp_change_stats_diag_zero[np.isin(np.arange(1985, 2022, 1), (array_year_plot[0:-1])), :, :]

        ##
        print(f'{state_name} IS area change stats summary:')
        # print_is_reduction_recovery(df_is_change_sum_plot)

        ##
        title_1 = None
        title_2 = None

        figsize = (25, 7)
        xlim = None

        if state_short_name == 'CA':
            ylim_1 = (8300, 10500)
            ylim_2 = (-10, 175)
        elif state_short_name == 'TX':
            ylim_1 = (7000, 11000)
            ylim_2 = (-10, 200)

        output_flag = False
        output_filename = join(fr'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend',
                               f'{state_short_name}_IS_area_change.jpg')
        sns_style = 'white'
        legend_flag = False
        plot_flag = 'area'
        flag_highlight_2008 = False

        manuscript_is_area_plot(df_is_change_sum_plot,
                                title_1, title_2,
                                figsize=figsize,
                                xlim=xlim, ylim_1=ylim_1, ylim_2=ylim_2,
                                output_flag=output_flag, output_filename=output_filename,
                                sns_style=sns_style,
                                legend_flag=legend_flag,
                                plot_flag=plot_flag,
                                flag_highlight_2008=flag_highlight_2008)

        ##

        (x, y, results) = get_weight_regression_results(data_flag=data_flag,
                                                        isp_folder=isp_folder)

        df_is_change_sum_conus_plot_adjust = generate_adjusted_conus_is_change_dataframe(df_is_change_sum_plot,
                                                                                         results,
                                                                                         img_sum_isp_change_stats_diag_zero_plot)

        print_adjust_is_area_change_stats(df_is_change_sum_conus_plot_adjust=df_is_change_sum_conus_plot_adjust)
        print('-------------------')

        ##
        output_filename_adjust = join(fr'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend',
                               f'{state_short_name}_IS_area_change_adjust.jpg')

        if state_short_name == 'CA':
            ylim_1 = (11000, 14400)
            ylim_2 = (-10, 175)
        elif state_short_name == 'TX':
            ylim_1 = (10600, 16500)
            ylim_2 = (-10, 220)

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


