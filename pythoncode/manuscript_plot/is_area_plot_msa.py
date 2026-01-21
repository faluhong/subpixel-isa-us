"""
    plot the selected Metropolitan (MSA) IS area change for the manuscript

    Current selection MSAs are: Atlanta-Sandy Springs-Roswell, GA  and  Boston-Cambridge-Newton, MA-NH
"""

import numpy as np
import os
from os.path import join, exists
import sys
import pandas as pd

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from analysis.utils_isp_change_stats_analysis import (generate_isp_change_summary_dataframe,)
from conus_isp_analysis.msa_is_change_display import (read_isp_change_stats_output_file_single_msa_year)
from pythoncode.manuscript_plot.utils_manuscript_is_area_plot import manuscript_is_area_plot

from uncertainty_estimation.is_area_adjustment_reg_all_sample import (get_weight_regression_results,)
from uncertainty_estimation.is_change_adjustment_reg import (generate_adjusted_conus_is_change_dataframe)
from uncertainty_estimation.print_adjust_is_area import (print_adjust_is_area_change_stats)

# def main():
if __name__ == '__main__':

    path_msa_2015 = join(rootpath, 'data', 'urban_pulse', 'shapefile', 'cb_2015_us_cbsa_500k')
    df_conus_msa_basic_info_drop_geometry = pd.read_csv(join(path_msa_2015, 'cb_2015_us_cbsa_500k_ard_conus.csv'))

    ##
    data_flag = 'conus_isp'  # 'conus_isp' or 'annual_nlcd'
    rootpath_nlcd_directory = None
    nlcd_folder = 'NLCD_annual'
    nlcd_filter_ocean_flag = False

    flag_mask_micropolitan = True

    # data_flag = 'conus_isp'
    rootpath_conus_isp = None
    # isp_folder = 'individual_year_tile_post_processing_is_expansion_ndvi015_sm'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    if data_flag == 'conus_isp':
        array_target_year = np.arange(1985, 2022, 1)
        output_path_figure = join(rootpath, 'results', 'isp_change_stats', 'msa_level', f'{data_flag}_figure', isp_folder)

    elif data_flag == 'annual_nlcd':
        # array_target_year = np.arange(1985, 2022, 1)
        array_target_year = np.arange(1985, 2023, 1)
        output_path_figure = join(rootpath, 'results', 'isp_change_stats', 'msa_level', f'{data_flag}_figure')
    else:
        raise ValueError('The data_flag is not correct')

    if flag_mask_micropolitan:
        df_conus_msa_basic_info_drop_geometry = df_conus_msa_basic_info_drop_geometry[df_conus_msa_basic_info_drop_geometry['LSAD'] == 'M1'].copy()
        df_conus_msa_basic_info_drop_geometry = df_conus_msa_basic_info_drop_geometry.reset_index(drop=True)
        print(f'After masking the micropolitan areas, the number of MSA is {len(df_conus_msa_basic_info_drop_geometry)}')

    # for i_county in range(0, len(df_conus_msa_basic_info_drop_geometry), 1):
    # for i_county in range(10, 10 + 1):
    for i_county in [6, 123]:   # Atlanta-Sandy Springs-Roswell, GA  and  Boston-Cambridge-Newton, MA-NH
        state_id = df_conus_msa_basic_info_drop_geometry['id'].values[i_county]

        msa_name = df_conus_msa_basic_info_drop_geometry['NAME'].values[i_county]
        # replace the '/' with '_' to avoid the error in the file name, such as Louisville/Jefferson County, KY-IN
        msa_name = msa_name.replace(r'/', '_')

        print(msa_name)

        if data_flag == 'conus_isp':
            output_path = join(rootpath, 'results', 'isp_change_stats', 'msa_level', 'conus_isp', f'{isp_folder}')
        elif data_flag == 'annual_nlcd':
            output_path = join(rootpath, 'results', 'isp_change_stats', 'msa_level', 'annual_nlcd')
        else:
            raise ValueError('The data_flag is not correct')

        output_filename = join(output_path, f'{msa_name}_isp_change_stats_{array_target_year[0]}_{array_target_year[-1]}.npy')
        if not exists(output_filename):
            print(f'Output file {output_filename} does not exist, skipping {msa_name}')
            continue
        # print(output_filename)

        (img_sum_isp_change_stats,
         img_sum_isp_change_stats_diag_zero) = read_isp_change_stats_output_file_single_msa_year(data_flag=data_flag,
                                                                                                 msa_name=msa_name,
                                                                                                 isp_folder=isp_folder,
                                                                                                 rootpath_conus_isp=None,
                                                                                                 rootpath_nlcd_directory=None)

        df_is_change_sum = generate_isp_change_summary_dataframe(img_sum_isp_change_stats, array_target_year)

        # array_plot_year = array_target_year
        array_year_plot = np.arange(1988, 2021, 1)
        df_is_change_sum_plot = df_is_change_sum[np.isin(df_is_change_sum['year_1'].values, (array_year_plot[0:-1]))].copy()

        img_sum_isp_change_stats_plot = img_sum_isp_change_stats[np.isin(np.arange(1985, 2022, 1), (array_year_plot[0:-1])), :, :]
        img_sum_isp_change_stats_diag_zero_plot = img_sum_isp_change_stats_diag_zero[np.isin(np.arange(1985, 2022, 1), (array_year_plot[0:-1])), :, :]

        print(f'{msa_name} IS area change stats summary:')
        # print_is_reduction_recovery(df_is_change_sum_plot)

        title_1 = None
        title_2 = None

        figsize = (25, 7)
        xlim = None
        ylim_1 = None
        ylim_2 = None

        output_flag = False
        output_filename = join(fr'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend',
                               f'{msa_name}_IS_area_change.jpg')
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

        output_filename_adjust = join(fr'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend',
                                      f'{msa_name}_IS_area_change_adjust.jpg')

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









