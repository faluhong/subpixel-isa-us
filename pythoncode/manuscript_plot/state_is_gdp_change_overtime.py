"""
    plot the relationship between IS area and state GDP over time for the manuscript
"""

import numpy as np
import os
from os.path import join
import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from matplotlib.ticker import FormatStrFormatter

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from conus_is_socio_economic.utils_prepare_ard_is_gdp_pop import (prepare_ard_state_is_gdp_population)


def manuscript_plot_is_area_gdp_change_overtime(array_x_plot,
                                                array_y_plot,
                                                array_target_year,
                                                region_name,
                                                cmap='viridis',
                                                ax_plot=None,
                                                x_label='IS area (km$^2$)',
                                                y_label='real GDP (billion $)',
                                                tick_label_size=28,
                                                axis_label_size=30,
                                                title_size=32,
                                                tick_length=4,
                                                scatter_size=100,
                                                figsize=(14, 8),
                                                vline_width=2.0,
                                                flag_plot_vline=True,
                                                x_scale='linear',
                                                y_scale='linear'):
    """
        Plot the IS area and GDP change over time
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

    ax_plot.set_xscale(x_scale)
    ax_plot.set_yscale(y_scale)

    # set the title
    title = f'{region_name}'

    ax_plot.set_title(title, size=title_size)
    plt.tight_layout()
    plt.show()


# def main():
if __name__ =='__main__':

    path_urban_pulse = join(rootpath, 'data', 'urban_pulse')

    # define the data flag and ISP folder, Annual NLCD ISP ranges from 1985 to 2023, CONUS ISP ranges from 1988 to 2020
    data_flag = 'conus_isp' # 'conus_isp' or 'annual_nlcd'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    flag_adjust = True
    print(f'Adjust IS area: {flag_adjust}')

    # keep the analysis period to from 1997 to 2020.
    # Because: (1) GDP data is available from 1997 to 2023, (2) Good-quality ISP data is available from 1988 to 2020
    array_target_year = np.arange(1997, 2021)

    (df_conus_state_basic_info,
     array_is_area,
     array_is_pct,
     array_real_gdp,
     array_nominal_gdp,
     array_pop) = prepare_ard_state_is_gdp_population(path_urban_pulse, data_flag, isp_folder, array_target_year,
                                                      flag_adjust=flag_adjust)

    print(data_flag, isp_folder)

    array_is_area_conus = np.nansum(array_is_area, axis=0)
    array_real_gdp_conus = np.nansum(array_real_gdp, axis=0)

    ##
    # plot the relationship between IS area and real GDP over time

    flag_plot_vline = True
    # super_x_label = f'IS area (km$^2$)'
    # super_y_label = 'real GDP (billion $)'
    super_x_label=None
    super_y_label=None
    x_scale = 'linear'
    y_scale = 'linear'
    cmap = 'viridis'

    plt.rcParams['font.family'] = "Arial"

    # sort the state by the name alphabetically
    df_conus_state_basic_info_sorted = df_conus_state_basic_info.sort_values(by='NAME')

    # fig, ax_plot = plt.subplots(ncols=7, nrows=7, figsize=(29, 22.5))
    fig, ax_plot = plt.subplots(ncols=5, nrows=10, figsize=(5.5 * 5, 4.0 * 10))

    manuscript_plot_is_area_gdp_change_overtime(array_x_plot=array_is_area_conus,
                                                array_y_plot=array_real_gdp_conus,
                                                array_target_year=array_target_year,
                                                region_name='CONUS',
                                                ax_plot=ax_plot[0, 0],
                                                cmap=cmap,
                                                x_label=None,
                                                y_label=None,
                                                tick_label_size=21,
                                                axis_label_size=19,
                                                title_size=24,
                                                scatter_size=50,
                                                flag_plot_vline=flag_plot_vline,
                                                x_scale=x_scale,
                                                y_scale=y_scale,
                                                )

    for i_state in range(0, len(df_conus_state_basic_info_sorted), 1):
        state_name = df_conus_state_basic_info_sorted['NAME'].values[i_state]

        index = df_conus_state_basic_info_sorted.index[i_state]

        array_gdp_ts = array_real_gdp[index, :]
        array_is_ts = array_is_area[index, :]

        row_index_plot = int((i_state + 1) / 5)
        col_index_plot = (i_state + 1) % 5

        manuscript_plot_is_area_gdp_change_overtime(array_x_plot=array_is_ts,
                                                    array_y_plot=array_gdp_ts,
                                                    array_target_year=array_target_year,
                                                    region_name=state_name,
                                                    ax_plot=ax_plot[row_index_plot, col_index_plot],
                                                    cmap=cmap,
                                                    x_label=None,
                                                    y_label=None,
                                                    tick_label_size=21,
                                                    axis_label_size=19,
                                                    title_size=24,
                                                    scatter_size=50,
                                                    flag_plot_vline=flag_plot_vline,
                                                    x_scale=x_scale,
                                                    y_scale=y_scale,
                                                    )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)

    fig.supxlabel(super_x_label, fontsize=20)
    fig.supylabel(super_y_label, fontsize=20)

    # plt.savefig(r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_GDP_trend\IS_area_vs_real_GDP_over_time.jpg',
    #             dpi=600, )
    # plt.close()
    plt.show()
