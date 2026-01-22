"""
    utility functions for plotting the IS area and GDP/Population change over time

    plot_is_area_gdp_change_overtime: plot the IS area and GDP change over time
    sum_plot_is_area_gdp_change_overtime: summary plot the IS area and GDP (Population) change over time for all the states
"""

import numpy as np
import os
from os.path import join, exists
import sys
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import matplotlib
from scipy import stats
from matplotlib.ticker import FormatStrFormatter

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def plot_is_area_gdp_change_overtime(array_x_plot,
                                     array_y_plot,
                                     array_target_year,
                                     region_name,
                                     cmap='viridis',
                                     ax_plot=None,
                                     x_label='IS area (km^2)',
                                     y_label='real GDP (billion $)',
                                     cbar_flag=False,
                                     tick_label_size=28,
                                     axis_label_size=30,
                                     title_size=32,
                                     tick_length=4,
                                     cbar_tick_label_size=20,
                                     scatter_size=100,
                                     figsize=(14, 8),
                                     vline_width=2.0,
                                     flag_plot_vline=True,
                                     x_scale='linear',
                                     y_scale='linear',
                                     title=None):
    """
        Plot the IS area and GDP change over time

    :param array_x_plot:
    :param array_y_plot:
    :param array_target_year:
    :param region_name:
    :param ax_plot:
    :param x_label:
    :param y_label:
    :param cbar_flag:
    :param tick_label_size:
    :param axis_label_size:
    :param title_size:
    :param tick_length:
    :param cbar_tick_label_size:
    :param scatter_size:
    :return:
    """

    matplotlib.rcParams['font.family'] = "Arial"

    if ax_plot is None:
        fig, ax_plot = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    img = ax_plot.scatter(array_x_plot, array_y_plot, label='real GDP', s=scatter_size, c=array_target_year, cmap=cmap)

    if flag_plot_vline:
        assert 2008 in array_target_year, '2008 is not in the GDP year list'
        ax_plot.axvline(x=array_x_plot[array_target_year == 2008][0],
                        color='black',
                        linestyle='dashed',
                        linewidth=vline_width)

    ax_plot.set_xlabel(x_label, size=axis_label_size)
    ax_plot.set_ylabel(y_label, size=axis_label_size)

    ax_plot.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    ax_plot.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

    if cbar_flag:
        # add the colorbar
        cb = plt.colorbar(img, cmap=cmap)
        cb.ax.tick_params(labelsize=cbar_tick_label_size)
        cb.ax.set_ylabel('Year', size=axis_label_size)

        cb.formatter = FormatStrFormatter('%d')  # Use '%d' for integer formatting

        cb.set_ticks(array_target_year)

        cb.update_ticks()

    ax_plot.set_xscale(x_scale)
    ax_plot.set_yscale(y_scale)

    # get the R2 value
    mask_valid = (~np.isnan(array_x_plot.ravel())) & (~np.isnan(array_y_plot.ravel()))

    if (x_scale == 'log') | (y_scale == 'log'):
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(array_x_plot.ravel())[mask_valid],
                                                                       np.log10(array_y_plot.ravel())[mask_valid])
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(array_x_plot.ravel()[mask_valid],
                                                                       array_y_plot.ravel()[mask_valid])

    r2 = r_value ** 2

    # set the title
    # if x_scale == 'log':
    #     title = f'N:{np.sum(mask_valid)}  $\\mathrm{{R^2}}$: {r2:.3f}   log(y) = {slope:.3f} log(x) + {intercept:.3f}'
    # else:
    #     title = f'N:{np.sum(mask_valid)} $\\mathrm{{R^2}}$: {r2:.3f}   y = {slope:.3f} x + {intercept:.3f}'
    # if x_scale == 'log':
    #     title = f'N:{np.sum(mask_valid)}  $\\mathrm{{R^2}}$: {r2:.3f}'
    # else:
    #     title = f'N:{np.sum(mask_valid)} $\\mathrm{{R^2}}$: {r2:.3f}'

    # set the title
    if title is None:
        title = f'{region_name}  $\\mathrm{{R^2}}$: {r2:.2f}'

    ax_plot.set_title(title, size=title_size)
    plt.tight_layout()
    plt.show()


def sum_plot_is_area_gdp_change_overtime(df_conus_state_basic_info,
                                         array_x_data,
                                         array_y_data,
                                         array_target_year,
                                         flag_plot_vline=True,
                                         super_x_label='Super X-axis Label',
                                         super_y_label='Super Y-axis Label',
                                         x_scale='linear',
                                         y_scale='linear',
                                         cmap='viridis',
                                         ):
    """
        summary plot the IS area and GDP (Population) change over time for all the states

    :param df_conus_state_basic_info:
    :param array_y_data:
    :param array_x_data:
    :param array_target_year:
    :param flag_plot_vline:
    :param super_x_label:
    :param super_y_label:
    :return:
    """

    plt.rcParams['font.family'] = "Arial"

    # sort the state by the name alphabetically
    df_conus_state_basic_info_sorted = df_conus_state_basic_info.sort_values(by='NAME')

    fig, ax_plot = plt.subplots(ncols=7, nrows=7, figsize=(30, 18))

    for i_state in range(0, len(df_conus_state_basic_info_sorted), 1):
        state_name = df_conus_state_basic_info_sorted['NAME'].values[i_state]

        index = df_conus_state_basic_info_sorted.index[i_state]

        array_gdp_ts = array_y_data[index, :]
        array_is_ts = array_x_data[index, :]

        plot_is_area_gdp_change_overtime(array_x_plot=array_is_ts,
                                         array_y_plot=array_gdp_ts,
                                         array_target_year=array_target_year,
                                         region_name=state_name,
                                         ax_plot=ax_plot[int(i_state / 7), i_state % 7],
                                         cmap=cmap,
                                         x_label=None,
                                         y_label=None,
                                         cbar_flag=False,
                                         tick_label_size=18,
                                         axis_label_size=16,
                                         title_size=21,
                                         scatter_size=50,
                                         flag_plot_vline=flag_plot_vline,
                                         x_scale=x_scale,
                                         y_scale=y_scale,
                                         )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)

    fig.supxlabel(super_x_label, fontsize=20)
    fig.supylabel(super_y_label, fontsize=20)

    plt.show()

