"""
    utility function to plot the IS area change for manuscript
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

from analysis.utils_isp_change_stats_analysis import (plot_isp_change_type_ts, plot_is_area_ts)


def manuscript_is_area_plot(df_is_change_sum_conus_plot, title_1, title_2, figsize=(24, 18),
                            xlim=None, ylim_1=None, ylim_2=None,
                            output_flag=False, output_filename=None,
                            sns_style='white',
                            legend_flag=True,
                            plot_flag='area',
                            flag_highlight_2008=False,
                            flag_adjust_with_ci=False,
                            fill_alpha=0.2,):
    """
        plot the IS area change for manuscript.
        Left figure is the IS area time series plot.
        Right figure is the IS area change type time series plot.

        :param df_is_change_sum_conus_plot:
        :param title_1:
        :param title_2:
        :param figsize:
        :param xlim:
        :param ylim_1:
        :param ylim_2:
        :param output_flag:
        :param output_filename:
        :param sns_style:
        :param legend_flag:
        :param plot_flag:
        :param flag_highlight_2008:
        :return:
    """

    sns.set_style(sns_style)
    figure_twin, axes_twin = plt.subplots(ncols=2, nrows=1, figsize=figsize)

    ax = plt.subplot(1, 2, 1)

    plot_is_area_ts(df_is_change_sum=df_is_change_sum_conus_plot,
                    title=title_1,
                    x_label='Year',
                    y_label='Area (km$^2$)',
                    y_label_right='Area percentage (%)',
                    x_axis_interval=2,
                    y_axis_interval=None,
                    flag_save=False,
                    output_filename=None,
                    axes=ax,
                    right_decimals=3,
                    figsize=(18, 10),
                    xlim=xlim,
                    ylim=ylim_1,
                    plot_flag=plot_flag,
                    legend_flag=legend_flag,
                    flag_highlight_2008=flag_highlight_2008,
                    flag_adjust_with_ci=flag_adjust_with_ci,
                    )

    ax = plt.subplot(1, 2, 2)

    if xlim is None:
        x_lim_isp_change = None
    else:
        x_lim_isp_change = (xlim[0] + 1, xlim[-1])

    plot_isp_change_type_ts(df_is_change_sum_conus_plot,
                            plot_flag=plot_flag,
                            title=title_2,
                            x_label='Year',
                            y_label='Area (km$^2$)',
                            y_label_right='Area percentage (%)',
                            x_axis_interval=2,
                            y_axis_interval=None,
                            flag_save=False,
                            output_filename=None,
                            axes=ax,
                            right_decimals=3,
                            figsize=(18, 10),
                            xlim=x_lim_isp_change,
                            ylim=ylim_2,
                            sns_style=sns_style,
                            legend_flag=legend_flag,
                            flag_highlight_2008=flag_highlight_2008,
                            flag_adjust_with_ci=flag_adjust_with_ci,
                            fill_alpha=fill_alpha,
                            )

    plt.tight_layout()

    if output_flag:
        assert output_filename is not None, 'output_filename is not provided'
        if not exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        plt.savefig(output_filename, dpi=300)
        plt.close()
    else:
        plt.show()

