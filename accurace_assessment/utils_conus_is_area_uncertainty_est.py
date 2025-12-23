"""
    utils to estimate the uncertainty of CONUS IS area estimation
"""

import numpy as np
import os
from os.path import join, exists
import sys
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import seaborn as sns
import matplotlib
import matplotlib.ticker as plticker
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from sample_based_analysis.conus_is_pct_accuracy_unbias import (prepare_evaluation_ard_data,
                                                                get_ns_is_weight,)

def get_conus_is_change_summary_data(output_folder, array_year_plot):
    """
        read the map-based CONUS IS pct and IS change stats
        :param data_flag:
        :param isp_folder:
        :return:
    """

    # load the change information from 1985 to 2022
    img_sum_isp_change_stats_conus = np.load(join(output_folder, 'conus_sum_isp_change_stats.npy'))
    img_sum_isp_change_stats_diag_zero_conus = np.load(join(output_folder, 'conus_sum_isp_change_stats_diag_zero.npy'))
    df_is_change_sum_conus = pd.read_csv(join(output_folder, 'conus_is_change_type_summary.csv'))

    # adjust to the years to be plotted
    df_is_change_sum_conus_plot = df_is_change_sum_conus[np.isin(df_is_change_sum_conus['year_1'].values, (array_year_plot[0:-1]))].copy()
    # img_sum_isp_change_stats_conus_plot = img_sum_isp_change_stats_conus[3:35]
    # img_sum_isp_change_stats_diag_zero_conus_plot = img_sum_isp_change_stats_diag_zero_conus[3:35]
    #
    img_sum_isp_change_stats_conus_plot = img_sum_isp_change_stats_conus[np.isin(np.arange(1985, 2022, 1), (array_year_plot[0:-1])), :, :]
    img_sum_isp_change_stats_diag_zero_conus_plot = img_sum_isp_change_stats_diag_zero_conus[np.isin(np.arange(1985, 2022, 1), (array_year_plot[0:-1])), :, :]

    return (df_is_change_sum_conus_plot,
            img_sum_isp_change_stats_conus_plot,
            img_sum_isp_change_stats_diag_zero_conus_plot,)


def read_sample_based_conus_is_pct(data_flag, isp_folder, array_target_year):
    """
        read the sample-based CONUS IS pct
        :param data_flag:
        :param isp_folder:
        :param array_target_year:
        :return:
    """

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
    array_sample_reference_isp = array_reference_isp_sum[0, :]
    array_sample_conus_isp = array_conus_isp_sum[0, :]

    # get the weight of different stratum
    (weight_ns_mean_conus, weight_is_mean_conus) = get_ns_is_weight(data_flag=data_flag,
                                                                    isp_folder=isp_folder,
                                                                    array_target_year=array_target_year)

    # sample-based IS pct using reference sample
    mask_natural = array_sample_conus_isp == 0
    sample_reference_isp = (weight_ns_mean_conus * np.nanmean(array_sample_reference_isp[mask_natural])
                            + weight_is_mean_conus * np.nanmean(array_sample_reference_isp[~mask_natural]))

    return (sample_reference_isp, array_sample_conus_isp, array_sample_reference_isp)


def plot_adjusted_are_with_uncertainty(df_is_change_sum,
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
                                       ):
    """
        plot adjusted area

        Note: 2025-12-09 The function is not recommended for usage.
                         Use the updated plot_is_area_ts and plot_isp_change_type_ts functions with flag_adjust_with_ci parameter instead.

        :param df_is_change_sum:
        :param title:
        :param x_label:
        :param y_label:
        :param y_label_right:
        :param x_axis_interval:
        :param y_axis_interval:
        :param right_decimals:
        :param figsize:
        :param xlim:
        :param ylim:
        :param legend_flag:
        :return:
    """

    sns.set_style("white")

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    legend_size = 24
    tick_label_size = 28
    axis_label_size = 30
    title_size = 32
    tick_length = 4

    line_width = 3.0
    linestyle = 'solid'

    axes.set_title(title, fontsize=title_size)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(2)
        axes.spines[axis].set_linewidth(2)

    array_year_plot = np.concatenate([df_is_change_sum['year_1'].values,
                                      np.array([df_is_change_sum['year_2'].values[-1]])])

    # plot the IS area changes
    array_is_area = np.concatenate([df_is_change_sum['is_area_year_1'].values, np.array([df_is_change_sum['is_area_year_2'].values[-1]])])
    array_is_area = array_is_area / 1000000  # convert the area to km^2

    axes.plot(array_year_plot, array_is_area, label='IS area', color='#363737',
              marker='o', markersize=14,
              linestyle=linestyle, linewidth=line_width)

    axes.fill_between(array_year_plot,
                      np.concatenate([df_is_change_sum['is_area_year_1_ci_lower'].values, np.array([df_is_change_sum['is_area_year_2_ci_lower'].values[-1]])]) / 1000000,
                      np.concatenate([df_is_change_sum['is_area_year_1_ci_upper'].values, np.array([df_is_change_sum['is_area_year_2_ci_upper'].values[-1]])]) / 1000000,
                      color='#A9A9A9',
                      alpha=0.5,
                      label='Uncertainty Range',
                      )

    axes.tick_params('x', labelsize=tick_label_size, direction='out',
                     length=tick_length, bottom=True, which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out',
                     length=tick_length, left=True, which='major')

    if x_axis_interval is not None:
        axes.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))
    if y_axis_interval is not None:
        axes.yaxis.set_major_locator(plticker.MultipleLocator(base=y_axis_interval))

    # Create a FuncFormatter to change the four-digit year to two-digit year
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x) % 100:02d}'))
    axes.ticklabel_format(style='plain', axis='y', useOffset=False)  # disable scientific notation

    axes.set_xlabel(x_label, size=axis_label_size)
    axes.set_ylabel(y_label, size=axis_label_size)

    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)

    # Remove duplicates by converting to a dictionary (preserves order)
    handles, labels = axes.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    if legend_flag:
        axes.legend(by_label.values(), by_label.keys(), loc='best', fontsize=legend_size)

    # add the second y-axis to show the percentage of the total area, the right y-axis
    ax_right = axes.secondary_yaxis('right')  # set the second y-axis, copy from the left y-axis
    ax_right.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    ax_right.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, right=True, which='major')

    ax_ticks = axes.get_yticks()  # get the ticks of the left y-axis
    ax_right_tick_labels = ax_ticks / (df_is_change_sum['total_area'].values[-1] / 1000000) * 100  # convert the area to percentage
    ax_right_tick_labels = np.round(ax_right_tick_labels, decimals=right_decimals)  # round the percentage to two decimal places

    # Use FixedLocator to ensure the tick labels are correctly aligned
    ax_right.yaxis.set_major_locator(FixedLocator(ax_ticks))
    ax_right.set_yticklabels(ax_right_tick_labels)  # set the right y-axis with the percentage label

    ax_right.set_ylabel(y_label_right, size=axis_label_size, labelpad=15)

    plt.tight_layout()
    plt.show()





