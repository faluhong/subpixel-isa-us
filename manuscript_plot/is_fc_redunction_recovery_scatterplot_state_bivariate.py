"""
    The script is to plot the bivariate scatterplot for the IS reduction rate vs IS recovery rate at the state level
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
from scipy import stats
from scipy.interpolate import interpn
from matplotlib.patches import Rectangle
from adjustText import adjust_text

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from Basic_tools.utils_hist_bar_plot import (hist_plot_stats)
from conus_isp_financial_crisis.is_fc_redunction_recovery_relationship import (get_overall_reduction_recovery_stats)


def separate_points_into_regions(array_reduction_rate, array_recovery_rate,
                                 array_threshold_reduction, array_threshold_recovery,
                                 array_color,):
    # separate the points into 9 regions based on the reduction and recovery thresholds
    df_point_region = pd.DataFrame(columns=['x', 'y', 'region'])

    rows = []

    for i_color in range(0, np.shape(array_color)[0]):
        for j_color in range(0, np.shape(array_color)[1]):

            if i_color == 0:
                mask_recovery = (array_recovery_rate <= array_threshold_recovery[i_color])
            elif i_color == np.shape(array_color)[0] - 1:
                mask_recovery = (array_recovery_rate > array_threshold_recovery[i_color - 1])
            else:
                mask_recovery = ((array_recovery_rate > array_threshold_recovery[i_color - 1])
                                 & (array_recovery_rate <= array_threshold_recovery[i_color]))

            if j_color == 0:
                mask_reduction = (array_reduction_rate <= array_threshold_reduction[j_color])
            elif j_color == np.shape(array_color)[1] - 1:
                mask_reduction = (array_reduction_rate > array_threshold_reduction[j_color - 1])
            else:
                mask_reduction = ((array_reduction_rate > array_threshold_reduction[j_color - 1])
                                  & (array_reduction_rate <= array_threshold_reduction[j_color]))

            mask_region = mask_reduction & mask_recovery

            x_plot = array_reduction_rate[mask_region]
            y_plot = array_recovery_rate[mask_region]

            # only add rows when there are points
            for xi, yi in zip(x_plot, y_plot):
                rows.append({'x': xi, 'y': yi, 'region': f'region_{i_color}_{j_color}'})

    df_point_region = pd.DataFrame(rows, columns=['x', 'y', 'region'])

    return df_point_region


# def main():
if __name__ == '__main__':

    ##
    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    # isp_folder = 'individual_year_tile_post_processing_mean_filter'

    flag_mask_micropolitan = True  # whether to mask out the Micropolitan in the MSA level analysis

    ##
    # for modify_target in ['state', 'msa', 'county']:
    for modify_target in ['state']:

        print(f'Processing {data_flag} with ISP folder: {isp_folder}, modify target: {modify_target}')

        output_filename = join(rootpath, 'results', 'isp_change_stats', f'{modify_target}_level', data_flag,
                               f'conus_{modify_target}_{isp_folder}_fc_impact.gpkg')
        gpd_annual_is = gpd.read_file(output_filename)

        if (modify_target == 'msa') & (flag_mask_micropolitan == True):
            # filter out the Micropolitan and keep the metropolitan
            gpd_annual_is = gpd_annual_is[gpd_annual_is['LSAD'] == 'M1'].copy()

        array_reduction_rate = gpd_annual_is['is_reduction_rate_fc'].values
        array_recovery_rate = gpd_annual_is['is_recovery_rate_fc'].values

        print(f'Mean IS reduction rate: {np.nanmean(array_reduction_rate):.4f} %')
        print(f'Mean IS recovery rate: {np.nanmean(array_recovery_rate):.4f} %')

        # overall reduction and recovery rate
        (is_reduction_rate, is_recovery_rate) = get_overall_reduction_recovery_stats(gpd_annual_is)
        print(f'Total IS reduction percentage: {is_reduction_rate:.4f} %')
        print(f'Total IS recovery percentage: {is_recovery_rate:.4f} %')

        # v3 colorbar
        # array_color = np.array([['#ffefe2', '#ffb286', '#f9752a', ],
        #                         ['#98cfe4', '#af978b', '#aa5f36', ],
        #                         ['#00afe8', '#427b8e', '#5b473c'],
        #                         ])

        # v4 colorbar
        array_color = np.array([['#e9e6f1', '#9ccae1', '#4fadcf',],
                                ['#e39bcb', '#9080be', '#3e64ad',],
                                ['#de50a6', '#833598', '#2b1a8a'],
                                ])

        # array_threshold_recovery = np.array([45.73034, 54.977084])
        # array_threshold_reduction = np.array([-61.2128, -47.175163])

        array_threshold_recovery = np.array([45.0, 55.0])
        array_threshold_reduction = np.array([-60.0, -45.0])

        df_point_region = separate_points_into_regions(array_reduction_rate, array_recovery_rate,
                                                      array_threshold_reduction=array_threshold_reduction,
                                                      array_threshold_recovery=array_threshold_recovery,
                                                      array_color=array_color,
                                                      )

        title = None
        flag_output = True
        output_filename = join(r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_reduction_recovery\v10',
                               f'IS_reduction_recovery_{modify_target}_level_bivariate.jpg')

        xlim = (-85, 0)
        ylim = (10, 100)
        flag_annotation_name = True
        list_annotation_name = gpd_annual_is['STUSPS'].values

        x_label = 'Reduction percentage (%)'
        y_label = 'Recovery percentage (%)'
        title = title
        figsize = (14, 14)
        dpi = 600

        sns.set_style("white")
        matplotlib.rcParams['font.family'] = "Arial"

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        tick_label_size = 34
        axis_label_size = 38
        cbar_tick_label_size = 24
        title_label_size = 32
        fonsize_annotation = 22
        tick_length = 8

        for spine in ax.spines.values():
            spine.set_linewidth(3.0)

        # plot the hlines and vlines
        for threshold_recovery in array_threshold_recovery:
            plt.hlines(y=threshold_recovery, xmin=xlim[0], xmax=xlim[1],
                       colors='#929591', linestyles='dashed', linewidth=2.5)
        for threshold_reduction in array_threshold_reduction:
            plt.vlines(x=threshold_reduction, ymin=ylim[0], ymax=ylim[1],
                       colors='#929591', linestyles='dashed', linewidth=2.5)

        for i_color in range(0, np.shape(array_color)[0]):
            for j_color in range(0, np.shape(array_color)[1]):

                mask_region = df_point_region['region'] == f'region_{i_color}_{j_color}'
                color_plot = array_color[i_color, j_color]
                x_plot = df_point_region.loc[mask_region, 'x'].values
                y_plot = df_point_region.loc[mask_region, 'y'].values

                img = plt.scatter(x_plot, y_plot, s=150, c=color_plot, edgecolors='black', linewidths=1.2)

        if flag_annotation_name:
            assert list_annotation_name is not None, 'Please provide the list of annotation names!'

            # Add all annotations first
            texts = []
            for i in range(len(list_annotation_name)):
                texts.append(ax.text(array_reduction_rate[i],
                                     array_recovery_rate[i],
                                     list_annotation_name[i],
                                     fontsize=fonsize_annotation,
                                     ha='center'))

            # Automatically adjust to prevent overlap
            adjust_text(texts, only_move={'texts': 'xy'},
                        # arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                        arrowprops=None,
                        )

        ax.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, width=2.5)
        ax.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, width=2.5)

        ax.set_xlabel(x_label, size=axis_label_size)
        ax.set_ylabel(y_label, size=axis_label_size)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_title(title, size=title_label_size)

        plt.tight_layout()

        if flag_output:
            assert output_filename is not None, 'Please provide the output filename!'

            if not exists(os.path.dirname(output_filename)):
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            plt.savefig(output_filename, dpi=dpi)
            plt.close()
        else:
            plt.show()