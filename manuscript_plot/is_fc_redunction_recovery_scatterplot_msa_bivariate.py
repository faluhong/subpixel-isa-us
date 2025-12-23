"""
    plot the bivariate scatterplot for the IS reduction rate vs IS recovery rate at the MSA level for the manuscript

    The top 10 population MSAs are highlighted in the plot.

    IS impact and recovery after the 2008 financial crisis

    Impact of the 2008 financial crisis:
    IS increase rate (%/year) three years before and after 2008 (2005-2008, 2008-2011)

    Recovery of the 2008 financial crisis:
    IS increase rate 2017-2020 / IS increase rate 2005-2008
"""

import numpy as np
import os
from os.path import join, exists
import sys
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import matplotlib
import seaborn as sns
import geopandas as gpd
from scipy import stats
from adjustText import adjust_text

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from conus_isp_financial_crisis.is_fc_redunction_recovery_relationship import (get_overall_reduction_recovery_stats)
from manuscript_plot.is_fc_redunction_recovery_scatterplot_state_bivariate import (separate_points_into_regions)
from conus_isp_financial_crisis.is_redunction_recovery_msa_size import (read_msa_gdp_population)
from conus_is_socio_economic.utils_prepare_ard_is_gdp_pop import (extract_msa_population)


# def main():
if __name__ =='__main__':

    ##
    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    flag_mask_micropolitan = True   # whether to mask out the Micropolitan in the MSA level analysis

    # for modify_target in ['state', 'msa', 'county']:
    for modify_target in ['msa']:

        print(f'Processing {data_flag} with ISP folder: {isp_folder}, modify target: {modify_target}')

        output_filename = join(rootpath, 'results', 'isp_change_stats', f'{modify_target}_level', data_flag,
                                f'conus_{modify_target}_{isp_folder}_fc_impact.gpkg')
        gpd_annual_is = gpd.read_file(output_filename)

        if (modify_target == 'msa') & (flag_mask_micropolitan == True):
            # filter out the Micropolitan and keep the metropolitan
            gpd_annual_is = gpd_annual_is[gpd_annual_is['LSAD'] == 'M1'].copy()

        array_reduction_rate = gpd_annual_is['is_reduction_rate_fc'].values
        array_recovery_rate = gpd_annual_is['is_recovery_rate_fc'].values
        # array_resilience_rate = array_recovery_rate - array_reduction_rate
        array_resilience_rate = array_recovery_rate + array_reduction_rate

        print(f'Mean IS reduction rate: {np.nanmean(array_reduction_rate):.4f} %')
        print(f'Mean IS recovery rate: {np.nanmean(array_recovery_rate):.4f} %')

        # overall reduction and recovery rate
        (is_reduction_rate, is_recovery_rate) = get_overall_reduction_recovery_stats(gpd_annual_is)
        print(f'Total IS reduction percentage: {is_reduction_rate:.4f} %')
        print(f'Total IS recovery percentage: {is_recovery_rate:.4f} %')

        # get the MSA population data

        (df_conus_msa_basic_info, df_msa_gdp, df_msa_pop) = read_msa_gdp_population()
        df_conus_msa_basic_info = df_conus_msa_basic_info[df_conus_msa_basic_info['LSAD'] == 'M1'].copy()

        array_target_year = np.arange(2001, 2021)

        (array_return_year, array_pop) = extract_msa_population(df_conus_msa_basic_info,
                                                                df_msa_pop,
                                                                array_target_year=array_target_year)
        # v4 colorbar
        array_color = np.array([['#e9e6f1', '#9ccae1', '#4fadcf', ],
                                ['#e39bcb', '#9080be', '#3e64ad', ],
                                ['#de50a6', '#833598', '#2b1a8a'],
                                ])

        # array_threshold_recovery = np.array([42.281396, 60.392537])
        # array_threshold_reduction = np.array([-63.953744, -48.358947])

        array_threshold_recovery = np.array([40.0, 60.0])
        array_threshold_reduction = np.array([-65.0, -50.0])

        df_point_region = separate_points_into_regions(array_reduction_rate, array_recovery_rate,
                                                       array_threshold_reduction=array_threshold_reduction,
                                                       array_threshold_recovery=array_threshold_recovery,
                                                       array_color=array_color,
                                                       )

        # xlim = (-100, 25)
        # ylim = (0, 125)
        # xlim = (-100, 85)
        # ylim = (-250, 210)
        xlim = (-100, 115)
        ylim = (-20, 215)
        output_filename = join(r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_reduction_recovery\v10',
                               f'IS_reduction_recovery_{modify_target}_level_bivariate.jpg')

        x_label='Reduction percentage (%)'
        y_label='Recovery percentage (%)'
        title = None
        figsize = (14, 14)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        tick_label_size = 34
        axis_label_size = 38
        title_label_size = 32
        tick_length = 8

        sns.set_style("white")
        matplotlib.rcParams['font.family'] = "Arial"

        for spine in ax.spines.values():
            spine.set_linewidth(3.0)

        # for i_color in range(0, np.shape(array_color)[0]):
        #     for j_color in range(0, np.shape(array_color)[1]):
        #         mask_region = df_point_region['region'] == f'region_{i_color}_{j_color}'
        #         color_plot = array_color[i_color, j_color]
        #         x_plot = df_point_region.loc[mask_region, 'x'].values
        #         y_plot = df_point_region.loc[mask_region, 'y'].values
        #
        #         img = plt.scatter(x_plot, y_plot, s=120, c=color_plot, edgecolors='black', linewidths=1.2)

        # plot the hlines and vlines
        for threshold_recovery in array_threshold_recovery:
            plt.hlines(y=threshold_recovery, xmin=xlim[0], xmax=xlim[1],
                       colors='#929591', linestyles='dashed', linewidth=2.5)
        for threshold_reduction in array_threshold_reduction:
            plt.vlines(x=threshold_reduction, ymin=ylim[0], ymax=ylim[1],
                       colors='#929591', linestyles='dashed', linewidth=2.5)

        # plot for the top 10 population MSAs
        # get the top 10 population MSAs
        list_annotation_name = gpd_annual_is['NAME'].values
        array_pop_2020 = array_pop[:, -1]
        array_pop_2020[np.isnan(array_pop_2020)] = 0  # set the NaN to 0
        index_sort_population = np.argsort(array_pop_2020)[::-1]
        # list_annotation_name_top10 = list_annotation_name[index_sort[0:10]]

        for i_index_sort in range(1, 11):
            index = index_sort_population[i_index_sort - 1]
            # index_top_ten_label = f'{i_index_sort}'

            reduction_rate_value = array_reduction_rate[index]
            recovery_rate_value = array_recovery_rate[index]
            annotation_name = list_annotation_name[index]

            mask_region = (df_point_region['x'] == reduction_rate_value) & (df_point_region['y'] == recovery_rate_value)
            assert np.sum(mask_region) == 1, 'The point for annotation is not unique!'

            color_plot = df_point_region['region'].values[mask_region][0]
            (i_color, j_color) = color_plot.split('_')[1:3]
            i_color = int(i_color)
            j_color = int(j_color)

            plt.scatter(reduction_rate_value, recovery_rate_value, s=150,
                        c=array_color[i_color, j_color], edgecolors='black', linewidths=1.2)

        texts = []

        # get the top 5 most resilient MSAs
        index_sort_resilience = np.argsort(array_resilience_rate)[::-1]
        for i_index_sort in range(1, 6):
            index = index_sort_resilience[i_index_sort - 1]

            reduction_rate_value = array_reduction_rate[index]
            recovery_rate_value = array_recovery_rate[index]
            annotation_name = list_annotation_name[index]

            texts.append(ax.text(reduction_rate_value,
                                 recovery_rate_value,
                                 annotation_name,
                                 fontsize=22,
                                 ha='center'))

            mask_region = (df_point_region['x'] == reduction_rate_value) & (df_point_region['y'] == recovery_rate_value)
            assert np.sum(mask_region) == 1, 'The point for annotation is not unique!'

            color_plot = df_point_region['region'].values[mask_region][0]
            (i_color, j_color) = color_plot.split('_')[1:3]
            i_color = int(i_color)
            j_color = int(j_color)

            plt.scatter(reduction_rate_value, recovery_rate_value, s=150,
                        c=array_color[i_color, j_color], edgecolors='black', linewidths=1.2)

        # plot the top 5 least resilient MSAs
        index_sort_resilience = np.argsort(array_resilience_rate)
        for i_index_sort in range(2, 7):    # The first one Chico, CA was not plotted due the 2018 Camp Fire impact
            index = index_sort_resilience[i_index_sort - 1]

            reduction_rate_value = array_reduction_rate[index]
            recovery_rate_value = array_recovery_rate[index]
            annotation_name = list_annotation_name[index]
            print(i_index_sort, annotation_name)

            texts.append(ax.text(reduction_rate_value,
                                 recovery_rate_value,
                                 annotation_name,
                                 fontsize=22,
                                 ha='center'))

            mask_region = (df_point_region['x'] == reduction_rate_value) & (df_point_region['y'] == recovery_rate_value)
            assert np.sum(mask_region) == 1, 'The point for annotation is not unique!'

            color_plot = df_point_region['region'].values[mask_region][0]
            (i_color, j_color) = color_plot.split('_')[1:3]
            i_color = int(i_color)
            j_color = int(j_color)

            plt.scatter(reduction_rate_value, recovery_rate_value, s=150,
                        c=array_color[i_color, j_color], edgecolors='black', linewidths=1.2)

        flag_annotation_name = True
        if flag_annotation_name:
            assert list_annotation_name is not None, 'Please provide the list of annotation names!'

            # Automatically adjust to prevent overlap
            adjust_text(texts, only_move={'texts': 'xy'},
                        # arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                        arrowprops=None,
                        )

        ax.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, width=2.0)
        ax.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, width=2.0)

        ax.set_xlabel(x_label, size=axis_label_size)
        ax.set_ylabel(y_label, size=axis_label_size)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.tight_layout()
        plt.show()

        # plt.savefig(output_filename, dpi=600)
        # plt.close()
#
