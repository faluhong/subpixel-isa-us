"""
    analyze the slope changes between IS area and GDP before and after the 2008 financial crisis
"""

import numpy as np
import os
from os.path import join
import sys
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import wilcoxon
from adjustText import adjust_text
import matplotlib.ticker as plticker
import pandas as pd

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.conus_isa_socio_economic.utils_prepare_ard_is_gdp_pop import (prepare_ard_state_is_gdp_population,
                                                                              prepare_ard_msa_is_gdp_population)


def calculate_pre_post_crisis_slope(df_target_basic_info, array_is_area, array_real_gdp, array_target_year):
    # calculate the slope between IS area and GDP before and after the financial crisis

    for i_target in range(0, len(df_target_basic_info)):
        target_name = df_target_basic_info['NAME'].values[i_target]
        print(f'Processing: {i_target} {target_name} ...')

        is_area_state = array_is_area[i_target, :]
        real_gdp_state = array_real_gdp[i_target, :]

        is_area_pre_crisis = is_area_state[array_target_year <= 2008]
        gdp_pre_crisis = real_gdp_state[array_target_year <= 2008]

        is_area_post_crisis = is_area_state[array_target_year > 2008]
        gdp_post_crisis = real_gdp_state[array_target_year > 2008]

        slope_pre, intercept_pre, r_value_pre, p_value_pre, std_err_pre = stats.linregress(is_area_pre_crisis,
                                                                                           gdp_pre_crisis, )

        slope_post, intercept_post, r_value_post, p_value_post, std_err_post = stats.linregress(is_area_post_crisis,
                                                                                                gdp_post_crisis, )

        # calculate the Thiel-Sen estimator as a robust check
        theil_slope_pre, intercept_pre, lower_pre, upper_pre = stats.theilslopes(gdp_pre_crisis, is_area_pre_crisis, 0.95)
        theil_slope_post, intercept_post, lower_post, upper_post = stats.theilslopes(gdp_post_crisis, is_area_post_crisis, 0.95)

        # print(f'State: {target_name}, Slope pre-crisis: {slope_pre:.6f}, Slope post-crisis: {slope_post:.6f}')

        df_target_basic_info.loc[i_target, 'slope_pre_crisis'] = slope_pre
        df_target_basic_info.loc[i_target, 'slope_post_crisis'] = slope_post

        df_target_basic_info.loc[i_target, 'theil_slope_pre_crisis'] = theil_slope_pre
        df_target_basic_info.loc[i_target, 'theil_slope_post_crisis'] = theil_slope_post

    df_target_basic_info['delta_slope'] = df_target_basic_info['slope_post_crisis'] - df_target_basic_info['slope_pre_crisis']
    df_target_basic_info['delta_theil_slope'] = df_target_basic_info['theil_slope_post_crisis'] - df_target_basic_info['theil_slope_pre_crisis']

    return df_target_basic_info


def calculate_conus_pre_post_crisis_slope(array_is_area_conus, array_real_gdp_conus, array_target_year_state):
    """
        Calculate the slope between IS area and GDP for CONUS before and after the financial crisis

        :param array_is_area_conus:
        :param array_real_gdp_conus:
        :param array_target_year_state:
        :return:
    """

    # calculate the percent IS area increase per year for CONUS
    is_area_pct_change_conus = (array_is_area_conus[-1] - array_is_area_conus[0]) / array_is_area_conus[0] * 100 / (array_target_year_state[-1] - array_target_year_state[0])
    print(f'CONUS IS area percent change per year: {is_area_pct_change_conus:.4f} %')
    # calculate the percent real GDP increase per year for CONUS
    gdp_pct_change_conus = (array_real_gdp_conus[-1] - array_real_gdp_conus[0]) / array_real_gdp_conus[0] * 100 / (array_target_year_state[-1] - array_target_year_state[0])
    print(f'CONUS real GDP percent change per year: {gdp_pct_change_conus:.4f} %')

    is_area_pre_crisis = array_is_area_conus[array_target_year_state <= 2008]
    gdp_pre_crisis = array_real_gdp_conus[array_target_year_state <= 2008]

    is_area_post_crisis = array_is_area_conus[array_target_year_state > 2008]
    gdp_post_crisis = array_real_gdp_conus[array_target_year_state > 2008]

    theil_slope_pre, intercept_pre, lower_pre, upper_pre = stats.theilslopes(gdp_pre_crisis, is_area_pre_crisis, 0.95)
    theil_slope_post, intercept_post, lower_post, upper_post = stats.theilslopes(gdp_post_crisis, is_area_post_crisis, 0.95)

    return (theil_slope_pre, theil_slope_post)


def plot_pre_after_crisis_is_gdp_slope(x_plot,
                                       y_plot,
                                       xlim, ylim,
                                       title=None,
                                       flag_annotation_name=False,
                                       list_annotation_name=None):

    x_label = 'Pre-crisis IS-GDP slope (Billion$/km²)'
    y_label = 'Post-crisis IS-GDP slope (Billion$/km²)'

    figsize = (14, 14)
    dpi = 600

    sns.set_style("white")
    matplotlib.rcParams['font.family'] = "Arial"

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    tick_label_size = 30
    axis_label_size = 33
    cbar_tick_label_size = 20
    title_label_size = 28
    fonsize_annotation = 18
    tick_length = 8

    for spine in ax.spines.values():
        spine.set_linewidth(3.0)

    # x_plot = df_state_basic_info['slope_pre_crisis'].values
    # y_plot = df_state_basic_info['slope_post_crisis'].values
    color_plot = 'skyblue'

    img = plt.scatter(x_plot, y_plot, s=150, c=color_plot, edgecolors='black', linewidths=1.2)

    # plot the 1:1 line
    ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], color='gray', linestyle='dashed', linewidth=2.5)

    if flag_annotation_name:
        assert list_annotation_name is not None, 'Please provide the list of annotation names!'

        # Add all annotations first
        texts = []
        for i in range(len(list_annotation_name)):
            texts.append(ax.text(x_plot[i],
                                 y_plot[i],
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

    ax.set_title(title, size=title_label_size)

    plt.tight_layout()
    plt.show()


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
    tick_label_size = 33
    axis_label_size = 35
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

    path_urban_pulse = join(rootpath, 'data', 'socio_economic')
    flag_adjust = True
    print(f'Adjust IS flag: {flag_adjust}')

    # define the data flag and ISP folder, Annual NLCD ISP ranges from 1985 to 2023, CONUS ISP ranges from 1988 to 2020
    data_flag = 'conus_isp'  # 'conus_isp' or 'annual_nlcd'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    array_target_year_state = np.arange(1997, 2021)
    array_target_year_msa = np.arange(2001, 2021)

    (df_state_basic_info,
     array_is_area_state,
     array_is_pct_state,
     array_real_gdp_state,
     array_nominal_gdp_state,
     array_pop_state) = prepare_ard_state_is_gdp_population(path_urban_pulse,
                                                            data_flag,
                                                            isp_folder,
                                                            array_target_year_state,
                                                            flag_adjust=flag_adjust)

    df_state_basic_info['pop_2020'] = array_pop_state[:, -1]

    ##
    (df_msa_basic_info,
     array_is_area_msa,
     array_is_pct_msa,
     array_real_gdp_msa,
     array_nominal_gdp_msa,
     array_pop_msa) = prepare_ard_msa_is_gdp_population(path_urban_pulse,
                                                        data_flag,
                                                        isp_folder,
                                                        array_target_year_msa,
                                                        flag_adjust=flag_adjust)

    df_msa_basic_info['pop_2020'] = array_pop_msa[:, -1]

    ##
    # aggregate IS area and GDP to CONUS scale

    array_is_area_conus = np.nansum(array_is_area_state, axis=0)
    array_real_gdp_conus = np.nansum(array_real_gdp_state, axis=0)

    (theil_slope_pre, theil_slope_post) = calculate_conus_pre_post_crisis_slope(array_is_area_conus,
                                                                               array_real_gdp_conus,
                                                                               array_target_year_state)

    print(f'CONUS Theil-Sen Slope pre-crisis: {theil_slope_pre:.6f}, post-crisis: {theil_slope_post:.6f}')
    print(f'CONUS Theil-Sen Slope change: {theil_slope_post - theil_slope_pre:.6f}')


    ##
    # plot the spatial distribution of slope changes between IS area and GDP before and after the financial crisis
    df_state_basic_info = calculate_pre_post_crisis_slope(df_target_basic_info=df_state_basic_info,
                                                          array_is_area=array_is_area_state,
                                                          array_real_gdp=array_real_gdp_state,
                                                          array_target_year=array_target_year_state)

    df_msa_basic_info = calculate_pre_post_crisis_slope(df_target_basic_info=df_msa_basic_info,
                                                        array_is_area=array_is_area_msa,
                                                        array_real_gdp=array_real_gdp_msa,
                                                        array_target_year=array_target_year_msa)

    # insert the slope ratio column
    df_state_basic_info['ratio_slope'] = df_state_basic_info['slope_post_crisis'] / df_state_basic_info['slope_pre_crisis'] * 100
    df_msa_basic_info['ratio_slope'] = df_msa_basic_info['slope_post_crisis'] / df_msa_basic_info['slope_pre_crisis'] * 100

    ##
    # conduct Wilcoxon signed-rank test on Theil-Sen slope changes
    results_state = wilcoxon(x=df_state_basic_info['theil_slope_pre_crisis'].values,
                             y=df_state_basic_info['theil_slope_post_crisis'].values,
                             alternative='two-sided')
    print(f'state-level Theil-Sen slope change Wilcoxon test results:')
    print(results_state)

    results_msa = wilcoxon(x=df_msa_basic_info['theil_slope_pre_crisis'].values,
                            y=df_msa_basic_info['theil_slope_post_crisis'].values,
                           alternative='two-sided')
    print(f'MSA-level Theil-Sen slope change Wilcoxon test results:')
    print(results_msa)

    ##

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
                                                  x_label='Pre-crisis urbanization-GDP productivity (Billion$/km²)',
                                                  y_label='Post-crisis urbanization-GDP productivity (Billion$/km²)',
                                                  flag_annotation_name=False,
                                                  list_annotation_name=None,
                                                  output_flag=output_flag,
                                                  output_filename=join(output_path,
                                                                       'State_IS_GDP_slope_pre_post_crisis_adjust.jpg'),
                                                  dpi=600,
                                                  figsize=(13.5, 13),
                                                  x_axis_interval=0.5,
                                                  y_axis_interval=0.5, )

    manuscript_plot_pre_after_crisis_is_gdp_slope(df_point_region_msa,
                                                  array_color,
                                                  xlim=(-5.1, 20.5),
                                                  ylim=(-5.0, 20.5),
                                                  title=None,
                                                  x_label='Pre-crisis urbanization-GDP productivity (Billion$/km²)',
                                                  y_label='Post-crisis urbanization-GDP productivity (Billion$/km²)',
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



