"""
    plot the pre-crisis and post-crisis ISA-GDP slope for the manuscript
"""

import numpy as np
import os
from os.path import join
import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from adjustText import adjust_text
import geopandas as gpd
import matplotlib.ticker as plticker

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.utils_hist_bar_plot import hist_plot_stats
from conus_isp_financial_crisis.is_gdp_slope_changes import (plot_pre_after_crisis_is_gdp_slope)


def separate_points_into_regions(df_state_basic_info, array_threshold, array_color):
    """

        :param array_color:
        :param df_state_basic_info:
        :return:
    """
    rows = []

    for i_color in range(0, len(array_color)):
        if i_color == 0:
            mask_color = (df_state_basic_info['delta_theil_slope'].values <= array_threshold[i_color])
        elif i_color == len(array_color) - 1:
            mask_color = (df_state_basic_info['delta_theil_slope'].values > array_threshold[i_color - 1])
        else:
            mask_color = ((df_state_basic_info['delta_theil_slope'].values > array_threshold[i_color - 1])
                          & (df_state_basic_info['delta_theil_slope'].values <= array_threshold[i_color]))

        x_region = df_state_basic_info['theil_slope_pre_crisis'].values[mask_color]
        y_region = df_state_basic_info['theil_slope_post_crisis'].values[mask_color]

        for xi, yi in zip(x_region, y_region):
            rows.append({'x': xi,
                         'y': yi,
                         'region': f'region_{i_color}',
                         'color': array_color[i_color]}, )

    df_point_region = pd.DataFrame(rows, columns=['x', 'y', 'region', 'color'])

    return df_point_region


def manuscript_plot_pre_after_crisis_is_gdp_slope(df_point_region,
                                                  array_color,
                                                  xlim,
                                                  ylim,
                                                  title=None,
                                                  x_label = 'Pre-crisis ISA-GDP slope (Billion$/km²)',
                                                  y_label = 'Post-crisis ISA-GDP slope (Billion$/km²)',
                                                  flag_annotation_name=False,
                                                  list_annotation_name=None,
                                                  output_flag=False,
                                                  output_filename=None,
                                                  dpi=600,
                                                  figsize = (14, 14),
                                                  x_axis_interval=None,
                                                  y_axis_interval=None,):

    sns.set_style("white")
    matplotlib.rcParams['font.family'] = "Arial"

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    tick_label_size = 34
    axis_label_size = 37
    title_label_size = 28
    fonsize_annotation = 22
    tick_length = 8

    for spine in ax.spines.values():
        spine.set_linewidth(4.0)

    for i_color in range(0, len(array_color)):
        mask_region = df_point_region['region'] == f'region_{i_color}'
        color_plot = array_color[i_color]
        x_plot = df_point_region.loc[mask_region, 'x'].values
        y_plot = df_point_region.loc[mask_region, 'y'].values

        img = plt.scatter(x_plot, y_plot, s=200, c=color_plot, edgecolors='black', linewidths=1.2)

    # plot the 1:1 line
    ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], color='gray', linestyle='dashed', linewidth=2.5)

    if flag_annotation_name:
        assert list_annotation_name is not None, 'Please provide the list of annotation names!'

        # Add all annotations first
        texts = []
        for i in range(len(list_annotation_name)):
            texts.append(ax.text(df_point_region['x'].values[i],
                                 df_point_region['y'].values[i],
                                 list_annotation_name[i],
                                 fontsize=fonsize_annotation,
                                 ha='center'))

        # Automatically adjust to prevent overlap
        adjust_text(texts,
                    only_move={'points': 'y', 'text': 'y'},
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                    # arrowprops=None,
                    )

    ax.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, width=2.5)
    ax.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, width=2.5)

    ax.set_xlabel(x_label, size=axis_label_size)
    ax.set_ylabel(y_label, size=axis_label_size)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if x_axis_interval is not None:
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))
    if y_axis_interval is not None:
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=y_axis_interval))

    ax.set_title(title, size=title_label_size)

    plt.tight_layout()

    if output_flag:
        assert output_filename is not None, "Please provide the output filename!"
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, dpi=dpi)
        plt.close()
    else:
        plt.show()


# def main():
if __name__ =='__main__':

    path_urban_pulse = join(rootpath, 'data', 'urban_pulse')

    # define the data flag and ISP folder, Annual NLCD ISP ranges from 1985 to 2023, CONUS ISP ranges from 1988 to 2020
    data_flag = 'conus_isp'  # 'conus_isp' or 'annual_nlcd'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    flag_adjust = True
    print(f'Adjust IS flag: {flag_adjust}')

    output_filename_state = join(rootpath, 'results', 'conus_is_socio_economic', f'state_level',
                                 f'state_is_gdp_slope_pre_post_financial_crisis_adjust.gpkg')
    df_state_basic_info = gpd.read_file(output_filename_state)

    output_filename_msa = join(rootpath, 'results', 'conus_is_socio_economic', f'msa_level',
                               f'msa_is_gdp_slope_pre_post_financial_crisis_adjust.gpkg')
    df_msa_basic_info = gpd.read_file(output_filename_msa)

    ##
    # plot the pre-crisis and post-crisis IS-GDP relationship for selected states
    # flag_annotation_name = False
    # list_annotation_name = df_state_basic_info['STUSPS'].values

    array_threshold = np.array([-1, 0, 0.5, 1])
    array_color = np.array(['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'])

    df_point_region_state = separate_points_into_regions(df_state_basic_info, array_threshold, array_color)
    df_point_region_msa = separate_points_into_regions(df_msa_basic_info, array_threshold, array_color)

    output_flag = False
    output_path = r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_GDP_slope'

    manuscript_plot_pre_after_crisis_is_gdp_slope(df_point_region_state,
                                                  array_color,
                                                  xlim=(-0.6, 4.5),
                                                  ylim=(-0.6, 4.5),
                                                  title=None,
                                                  x_label='Pre-crisis urbanization-GDP intensity (Billion$/km²)',
                                                  y_label='Post-crisis urbanization-GDP intensity (Billion$/km²)',
                                                  flag_annotation_name=False,
                                                  list_annotation_name=None,
                                                  output_flag=output_flag,
                                                  output_filename=join(output_path,
                                                                       'State_IS_GDP_slope_pre_post_crisis_adjust.jpg'),
                                                  dpi=600,
                                                  figsize=(13.5, 13),
                                                  x_axis_interval=0.5,
                                                  y_axis_interval=0.5,)

    manuscript_plot_pre_after_crisis_is_gdp_slope(df_point_region_msa,
                                                  array_color,
                                                  xlim=(-5.1, 20.5),
                                                  ylim=(-5.0, 20.5),
                                                  title=None,
                                                  x_label='Pre-crisis urbanization-GDP intensity (Billion$/km²)',
                                                  y_label='Post-crisis urbanization-GDP intensity (Billion$/km²)',
                                                  flag_annotation_name=False,
                                                  list_annotation_name=None,
                                                  output_flag=output_flag,
                                                  output_filename=join(output_path,
                                                                       'MSA_IS_GDP_slope_pre_post_crisis_adjust.jpg'),
                                                  dpi=600,
                                                  figsize=(13.5, 13),
                                                  x_axis_interval=5,
                                                  y_axis_interval=5,
                                                  )

    ##

    # plot_pre_after_crisis_is_gdp_slope(x_plot=df_state_basic_info['theil_slope_pre_crisis'].values,
    #                                    y_plot=df_state_basic_info['theil_slope_post_crisis'].values,
    #                                    xlim=(-0.6, 5.0),
    #                                    ylim=(-0.6, 5.0),
    #                                    title=f'Pre- and Post-2008 Financial Crisis State-level IS-GDP, Theil Slope',
    #                                    flag_annotation_name=True,
    #                                    list_annotation_name=df_state_basic_info['STUSPS'].values)

    # x_plot = df_state_basic_info['theil_slope_pre_crisis'].values
    # y_plot = df_state_basic_info['theil_slope_post_crisis'].values
    # xlim = (-0.5, 4.5)
    # ylim = (-0.5, 4.5)
    # title = f'Pre- and Post-2008 Financial Crisis State-level IS-GDP, Theil Slope'
    # title = None
    # flag_annotation_name = False
    # list_annotation_name = None
    #
    # x_label = 'Pre-crisis ISA-GDP slope (Billion$/km²)'
    # y_label = 'Post-crisis ISA-GDP slope (Billion$/km²)'














