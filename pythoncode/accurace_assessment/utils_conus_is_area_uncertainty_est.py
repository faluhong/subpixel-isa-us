"""
    utils to estimate the uncertainty of CONUS IS area estimation
"""

import numpy as np
import os
from os.path import join
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


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








