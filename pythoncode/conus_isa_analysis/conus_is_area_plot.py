"""
    plot the CONUS impervious surface area change
"""

import numpy as np
import os
from os.path import join, exists
import sys
from osgeo import gdal, gdal_array, gdalconst

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.conus_isa_analysis.utils_plot_is_area import (manuscript_is_area_plot)
from pythoncode.conus_isa_financial_crisis_resilience.utils_print_red_rec import (print_is_reduction_recovery,
                                                                                  print_adjust_is_area_change_stats)

from pythoncode.accurace_assessment.is_change_adjustment_reg import (get_weight_regression_results,
                                                                     get_conus_is_change_summary_data,
                                                                     generate_adjusted_conus_is_change_dataframe)


# def main():
if __name__ =='__main__':

    ##
    data_flag = 'conus_isp'   # 'conus_isp' or 'annual_nlcd'

    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    title = 'CONUS ISP'

    output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag, isp_folder)



    print(data_flag, isp_folder, title)

    ##
    # define the time period for the plot
    array_year_plot = np.arange(1988, 2021, 1)

    (x, y, results) = get_weight_regression_results(data_flag=data_flag, isp_folder=isp_folder)

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
    print_adjust_is_area_change_stats(df_is_change_sum_conus_plot_adjust=df_is_change_sum_conus_plot_adjust)
    ##
    # plot the IS change

    title_1 = f'{title}: IS area change'
    title_2 = f'{title}: year-to-year IS area change'

    figsize = (25, 7)
    xlim = None
    ylim_1 = None
    ylim_2 = None

    sns_style = 'white'
    legend_flag = True
    plot_flag = 'area'
    flag_highlight_2008 = False

    # plot the mapped IS area change
    manuscript_is_area_plot(df_is_change_sum_conus_plot,
                            title_1, title_2,
                            figsize=figsize,
                            xlim=xlim, ylim_1=ylim_1, ylim_2=ylim_2,
                            output_flag=False,
                            output_filename=None,
                            sns_style=sns_style,
                            legend_flag=legend_flag,
                            plot_flag=plot_flag,
                            flag_highlight_2008=flag_highlight_2008,
                            flag_focus_on_growth_types=True)

    # plot the adjust the IS area with uncertainty interval
    manuscript_is_area_plot(df_is_change_sum_conus_plot_adjust,
                            title_1,
                            title_2,
                            figsize=figsize,
                            xlim=xlim, ylim_1=ylim_1, ylim_2=ylim_2,
                            output_flag=False,
                            output_filename=None,
                            sns_style=sns_style,
                            legend_flag=legend_flag,
                            plot_flag=plot_flag,
                            flag_highlight_2008=flag_highlight_2008,
                            flag_adjust_with_ci=True,
                            fill_alpha=0.3,
                            flag_focus_on_growth_types=True,
                            )












