"""
    Boxplot the relationship between the MSA size with the IS reduction and recovery percentage for the manuscript

    The MSA size is based on the population in 2020, which can also be determined by the GDP size.

    MSA size is divided into 4 categories:
        1: < 250 thousands
        2: 250 - 500 thousands
        3: 500 thousands - 1 million
        4: > 1 million
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
import geopandas as gpd

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from Basic_tools.utils_hist_bar_plot import (hist_plot_stats)
from conus_is_socio_economic.utils_prepare_ard_is_gdp_pop import (extract_msa_population)
from conus_isp_financial_crisis.is_redunction_recovery_msa_size import (plot_relationship_between_is_changes_with_gdp_population,
                                                                        read_msa_gdp_population,
                                                                        conduct_anova_test_pop)

def manuscript_boxplot_reduction_recovery_msa_size(df_analysis_ready,
                                                   tick_labels=None,
                                                   ax_plot=None,
                                                   fig_size=(16, 12),
                                                   x_attribute='population_category',
                                                   boxplot_alpha=1.0,
                                                   x_tick_label_size=28,
                                                   y_tick_label_size=28,
                                                   axis_label_size=32,
                                                   y_axis_scale='linear',
                                                   showfliers=True,
                                                   y_lim=None,
                                                   title='MSA population vs IS reduction rate',
                                                   x_label='MSA population (thousands in 2020)',
                                                   y_label='IS reduction rate (%)',
                                                   y_attribute='is_reduction_rate_fc',
                                                   legend_size=24,
                                                   output_flag=False,
                                                   output_filename=None, ):
    """
        plot the boxplot between the MSA size and the reduction/recovery rate

        :param ax_plot:
        :param fig_size:
        :param x_attribute:
        :param boxplot_alpha:
        :param x_tick_label_size:
        :param y_tick_label_size:
        :param axis_label_size:
        :param y_axis_scale:
        :param showfliers:
        :param y_lim:
        :param title:
        :param x_label:
        :param y_label:
        :param y_attribute:
        :param legend_size:
        :return:
    """
    sns.set_style(style="white")
    plt.rcParams['font.family'] = 'Arial'

    if ax_plot is None:
        figure, ax_plot = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    # x_tick_label_size = 28
    # y_tick_label_size = 28
    # axis_label_size = 32
    title_size = 26
    mean_point_marker_size = 18
    filter_point_marker_size = 16

    tick_length = 4
    axes_line_width = 2.0
    box_line_width = 3.0

    for spine in ax_plot.spines.values():
        spine.set_linewidth(axes_line_width)  # Set the desired width here

    sns.boxplot(data=df_analysis_ready,
                x=x_attribute,
                y=y_attribute,
                # hue=x_attribute,
                # order=tick_labels,
                boxprops=dict(linewidth=box_line_width,
                              alpha=boxplot_alpha,
                              facecolor='none'),
                showmeans=True,
                whiskerprops=dict(linestyle='-', linewidth=box_line_width),
                medianprops=dict(linewidth=box_line_width),
                capprops=dict(linewidth=box_line_width),
                meanprops=dict(markersize=mean_point_marker_size,
                               markeredgecolor='black',
                               markerfacecolor='black',
                               marker='o', ),
                showfliers=showfliers,
                fliersize=filter_point_marker_size,
                flierprops={"marker": "^", "markeredgecolor": "black", "markersize": 12, "markerfacecolor": "black"},
                width=0.5,
                dodge="auto",
                # palette=dict_palette,
                saturation=1.0,
                ax=ax_plot,
                )

    if tick_labels is None:
        tick_labels = ['< 250k', '250k-500k', '500k-1M', '> 1M']
    ax_plot.set_xticks(ticks=np.arange(0, len(tick_labels)), labels=tick_labels)

    ax_plot.set_yscale(y_axis_scale)

    ax_plot.tick_params('x', labelsize=x_tick_label_size, direction='out', length=tick_length, width=axes_line_width, top=False, which='major', rotation=0)
    ax_plot.tick_params('y', labelsize=y_tick_label_size, direction='out', length=tick_length, width=axes_line_width, left=True, which='major', rotation=0)

    ax_plot.set_xlabel(x_label, size=axis_label_size)
    ax_plot.set_ylabel(y_label, size=axis_label_size)

    ax_plot.set_ylim(y_lim)

    ax_plot.set_title(title, size=title_size)

    # ax_plot.legend(fontsize=legend_size)

    plt.tight_layout()

    if output_flag:
        if output_filename is None:
            raise ValueError('The output filename is not provided')

        if not exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        plt.savefig(output_filename, dpi=600)
        plt.close()
    else:
        plt.show()

# def main():
if __name__ =='__main__':

    ##
    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    # isp_folder = 'individual_year_tile_post_processing_mean_filter'

    modify_target = 'msa'     # 'state', 'msa', 'county'

    (df_conus_msa_basic_info, df_msa_gdp, df_msa_pop) = read_msa_gdp_population()

    print(f'Processing {data_flag} with ISP folder: {isp_folder}, modify target: {modify_target}')

    output_filename = join(rootpath, 'results', 'isp_change_stats', f'{modify_target}_level', data_flag,
                            f'conus_{modify_target}_{isp_folder}_fc_impact.gpkg')
    gpd_annual_is = gpd.read_file(output_filename)

    # keep the M1 metropolitan areas
    df_conus_msa_basic_info = df_conus_msa_basic_info[df_conus_msa_basic_info['LSAD'] == 'M1'].copy()
    gpd_annual_is = gpd_annual_is[gpd_annual_is['LSAD'] == 'M1'].copy()

    ##
    array_reduction_rate = gpd_annual_is['is_reduction_rate_fc'].values
    array_recovery_rate = gpd_annual_is['is_recovery_rate_fc'].values

    array_target_year = np.arange(2001, 2021)

    (array_return_year, array_pop) = extract_msa_population(df_conus_msa_basic_info,
                                                            df_msa_pop,
                                                            array_target_year=array_target_year)

    ##
    # remove the MSAs with all NaN population values
    mask_nan = np.isnan(array_pop).all(axis=1)

    array_pop = array_pop[~mask_nan, :]
    array_reduction_rate = array_reduction_rate[~mask_nan]
    array_recovery_rate = array_recovery_rate[~mask_nan]

    df_conus_msa_basic_info = df_conus_msa_basic_info[~mask_nan]
    gpd_annual_is = gpd_annual_is[~mask_nan]

    # prepare the dataframe for analysis
    df_analysis_ready = df_conus_msa_basic_info.copy()

    df_analysis_ready['is_reduction_rate_fc'] = array_reduction_rate
    df_analysis_ready['is_recovery_rate_fc'] = array_recovery_rate
    df_analysis_ready['population_2020'] = array_pop[:, -1]

    # df_analysis_ready.to_excel(join(rootpath, 'results', 'isp_change_stats', 'msa_level', 'conus_isp',
    #                                 f'conus_msa_M1_{isp_folder}_fc_impact_population.xlsx'), index=False)

    # divide the population into 4 categories
    array_population_category = np.zeros(np.shape(df_analysis_ready['population_2020'].values), dtype=int)
    array_population_category[array_pop[:, -1] < 250] = 1
    array_population_category[(array_pop[:, -1] >= 250) & (array_pop[:, -1] < 500)] = 2
    array_population_category[(array_pop[:, -1] >= 500) & (array_pop[:, -1] < 1000)] = 3
    array_population_category[array_pop[:, -1] >= 1000] = 4

    df_analysis_ready['population_category'] = array_population_category

    print(np.unique(array_population_category, return_counts=True))

    ##
    output_flag = False
    output_path = r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_reduction_recovery_MSA_size'

    manuscript_boxplot_reduction_recovery_msa_size(df_analysis_ready=df_analysis_ready,
                                                   ax_plot=None,
                                                   fig_size=(12, 10),
                                                   x_attribute='population_category',
                                                   boxplot_alpha=1.0,
                                                   x_tick_label_size=28,
                                                   y_tick_label_size=28,
                                                   axis_label_size=32,
                                                   y_axis_scale='linear',
                                                   showfliers=True,
                                                   y_lim=(0, 120),
                                                   # title='MSA population vs IS recovery rate',
                                                   title=None,
                                                   x_label='Population',
                                                   y_label='Recovery percentage (%)',
                                                   y_attribute='is_recovery_rate_fc',
                                                   legend_size=24,
                                                   output_flag=output_flag,
                                                   output_filename=join(output_path, 'boxplot_recovery.jpg'), )

    manuscript_boxplot_reduction_recovery_msa_size(df_analysis_ready=df_analysis_ready,
                                                   ax_plot=None,
                                                   fig_size=(12, 10),
                                                   x_attribute='population_category',
                                                   boxplot_alpha=1.0,
                                                   x_tick_label_size=28,
                                                   y_tick_label_size=28,
                                                   axis_label_size=32,
                                                   y_axis_scale='linear',
                                                   showfliers=True,
                                                   y_lim=(-100, 40),
                                                   title=None,
                                                   x_label='Population',
                                                   y_label='Reduction percentage (%)',
                                                   y_attribute='is_reduction_rate_fc',
                                                   legend_size=24,
                                                   output_flag=output_flag,
                                                   output_filename=join(output_path, 'boxplot_reduction.jpg'),
                                                   )

    ##
    conduct_anova_test_pop(df_analysis_ready=df_analysis_ready, attribute_name='is_reduction_rate_fc')
    conduct_anova_test_pop(df_analysis_ready=df_analysis_ready, attribute_name='is_recovery_rate_fc')

    ##













