"""
    analyze the IS impact and recovery after the 2008 financial crisis

    Impact of the 2008 financial crisis:
    IS increase rate (%/year) three years before and after 2008 (2005-2008, 2008-2011)

    Recovery of the 2008 financial crisis:
    IS increase rate 2017-2020 / IS increase rate 2005-2008
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import ScalarMappable

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

# from Basic_tools.Figure_plot import FP
# from Basic_tools.utils_hist_bar_plot import (hist_plot_stats)


def plot_is_geodataframe(gpd_annual_is,
                         column_name,
                         title=None,
                         vmin=None,
                         vmax=None,
                         annotation_flag=True,
                         annotation_size=20,
                         annotation_fmt='%.1f',
                         annotation_flag_ignore_dc=True,
                         flag_cbar=True,
                         cmap=plt.get_cmap('RdYlGn_r'),
                         fig_size=(18, 12),
                         ):
    """
        Plot the GeoDataFrame of annual IS data with specified column and title.
        Args:
            gpd_annual_is (GeoDataFrame): The GeoDataFrame containing annual IS data.
            column_name (str): The column name to plot.
            title (str): The title of the plot.
            vmin (float): Minimum value for color scaling.
            vmax (float): Maximum value for color scaling.
            annotation_flag (bool): Whether to annotate the polygons with values.
            annotation_size (int): Font size for annotations.
            annotation_fmt (str): Format string for annotations.
            annotation_flag_ignore_dc (bool): Whether to ignore DC in annotations due to its small area.
            flag_cbar (bool): Whether to display a colorbar.
            cmap (Colormap): Colormap to use for the plot.
            fig_size (tuple): Size of the figure.
    """
    fig, ax_plot = plt.subplots(figsize=fig_size)

    title_label_size = 25
    axis_tick_label_size = 18
    axis_label_size = 22

    cbar_tick_label_size = 22
    tick_length = 6
    axes_line_width = 1.5

    ticks_flag = True

    if vmin is None:
        vmin = np.nanmin(gpd_annual_is[column_name].values)
    if vmax is None:
        vmax = np.nanmax(gpd_annual_is[column_name].values)

    gpd_annual_is.boundary.plot(ax=ax_plot, color="black", linewidth=1)
    gpd_annual_is.plot(ax=ax_plot, column=column_name, cmap=cmap, legend=False,
                            vmin=vmin, vmax=vmax,)

    if annotation_flag:
        # Add annotation (e.g., label each polygon with its 'name' column)
        for idx, row in gpd_annual_is.iterrows():

            if annotation_flag_ignore_dc and row['NAME'] == 'District of Columbia':
                continue

            centroid = row['geometry'].representative_point()

            value = row[column_name]
            annotation_text = annotation_fmt % value
            ax_plot.text(centroid.x, centroid.y, annotation_text,
                         fontsize=annotation_size, ha='center', va='center')

    if flag_cbar:
        # define the colorbar
        divider = make_axes_locatable(ax_plot)

        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])  # Dummy array for colorbar

        cax = divider.append_axes("right", size="2%", pad=0.01)

        cb = plt.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=cbar_tick_label_size)

        # cb.ax.set_ylabel('Year', size=axis_label_size)
        # cb.formatter = FormatStrFormatter('%d')  # Use '%d' for integer formatting

    ax_plot.tick_params('x', labelsize=axis_tick_label_size, direction='out', length=tick_length,
                        width=axes_line_width, bottom=ticks_flag, labelbottom=ticks_flag, which='major')
    ax_plot.tick_params('y', labelsize=axis_tick_label_size, direction='out', length=tick_length,
                        width=axes_line_width, left=ticks_flag, labelleft=ticks_flag, which='major')

    ax_plot.set_title(title, size=title_label_size)
    ax_plot.set_xlabel("Longitude", size=axis_label_size)
    ax_plot.set_ylabel("Latitude", size=axis_label_size)

    plt.tight_layout()
    plt.show()


def calculate_fc_impact_recovery(gpd_annual_is_input):
    """
        calculate the reduction and recovery of IS increase rate due to the 2008 financial crisis

        2005-2008: pre-crisis increase rate
        2008-2011: post-crisis increase rate
        2017-2020: recovery increase rate

        :param gpd_annual_is:
        :return:
    """

    gpd_annual_is = gpd_annual_is_input.copy()

    for i_state in range(0, len(gpd_annual_is)):
        # calculate the increase percentage of IS area from 1988 to 2020
        is_area_1988 = gpd_annual_is.loc[i_state, 'is_area_1988']
        is_area_2020 = gpd_annual_is.loc[i_state, 'is_area_2020']

        is_area_increase_pct_1988_2020 = (is_area_2020 - is_area_1988) / is_area_1988 * 100

        gpd_annual_is.loc[i_state, 'is_area_increase_pct_1988_2020'] = is_area_increase_pct_1988_2020

        # calculate ISP increase rate between (2005-2008) and (2008-2011), and calculate the reduction percentage of ISP increase rate due to the 2008 financial crisis
        # impact of the 2008 financial crisis on the increase rate of IS percentage
        is_pct_2005 = gpd_annual_is.loc[i_state, 'is_pct_2005']
        is_pct_2008 = gpd_annual_is.loc[i_state, 'is_pct_2008']
        is_pct_2011 = gpd_annual_is.loc[i_state, 'is_pct_2011']

        is_increase_rate_2005_2008 = (is_pct_2008 - is_pct_2005) / 3
        is_increase_rate_2008_2011 = (is_pct_2011 - is_pct_2008) / 3

        is_reduction_rate = (is_increase_rate_2008_2011 - is_increase_rate_2005_2008) / is_increase_rate_2005_2008 * 100

        # recovery of IS increase
        is_pct_2017 = gpd_annual_is.loc[i_state, 'is_pct_2017']
        is_pct_2020 = gpd_annual_is.loc[i_state, 'is_pct_2020']

        is_increase_rate_2017_2020 = (is_pct_2020 - is_pct_2017) / 3

        is_recovery_rate = is_increase_rate_2017_2020 / is_increase_rate_2005_2008 * 100

        gpd_annual_is.loc[i_state, 'is_increase_rate_2005_2008'] = is_increase_rate_2005_2008
        gpd_annual_is.loc[i_state, 'is_increase_rate_2008_2011'] = is_increase_rate_2008_2011
        gpd_annual_is.loc[i_state, 'is_reduction_rate_fc'] = is_reduction_rate
        gpd_annual_is.loc[i_state, 'is_increase_rate_2017_2020'] = is_increase_rate_2017_2020
        gpd_annual_is.loc[i_state, 'is_recovery_rate_fc'] = is_recovery_rate

        gpd_annual_is.loc[i_state, 'if_resilience'] = is_recovery_rate - is_reduction_rate

    return gpd_annual_is


# def main():
if __name__ =='__main__':

    ##
    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    # isp_folder = 'individual_year_tile_post_processing_mean_filter'

    modify_target = 'state'     # 'state', 'msa', 'county'

    print(f'Processing {data_flag} with ISP folder: {isp_folder}, modify target: {modify_target}')

    if data_flag == 'conus_isp':
        filename_annual_is_gpkg = join(rootpath, 'results', 'isp_change_stats', f'{modify_target}_level', data_flag,
                               f'conus_{modify_target}_is_stats_{isp_folder}.gpkg')
    elif data_flag == 'annual_nlcd':
        filename_annual_is_gpkg = join(rootpath, 'results', 'isp_change_stats', f'{modify_target}_level', data_flag,
                               f'conus_{modify_target}_is_stats_{data_flag}.gpkg')
    else:
        raise ValueError('The data_flag is not correct')

    gpd_annual_is = gpd.read_file(filename_annual_is_gpkg)
    gpd_annual_is = gpd_annual_is.to_crs(epsg=4326)  # convert to WGS84

    gpd_annual_is = calculate_fc_impact_recovery(gpd_annual_is)

    ##
    # hist_plot_stats(gpd_annual_is['is_reduction_rate_fc'].values)
    # plot the histogram of IS recovery percentage from 1988 to 2020
    # hist_plot_stats(gpd_annual_is['is_recovery_rate_fc'].values, bins=np.arange(0, 150, 10))

    ## output the GeoDataFrame
    # output_filename = join(rootpath, 'results', 'isp_change_stats', f'{modify_target}_level', data_flag,
    #                         f'conus_{modify_target}_{isp_folder}_fc_impact.gpkg')
    # gpd_annual_is.to_file(output_filename, driver='GPKG')

    ##
    # plot the maps of IS percentage in 1988 and 2020, IS area increase percentage from 1988 to 2020,
    # reduction of IS increase rate due to the 2008 financial crisis, recovery of IS

    flag_cbar = True
    annotation_flag = False
    cmap = plt.get_cmap('RdYlGn_r')

    plot_is_geodataframe(gpd_annual_is,
                         column_name='is_pct_1988',
                         title=f'ISP in 1988',
                         vmin=0, vmax=4.0,
                         annotation_flag=annotation_flag, annotation_size=15, annotation_fmt='%.2f',
                         flag_cbar=flag_cbar, cmap=cmap,
                         fig_size=(18, 12))

    plot_is_geodataframe(gpd_annual_is,
                         column_name='is_pct_2020',
                         title=f'ISP in 2020',
                         vmin=0, vmax=4.0,
                         annotation_flag=annotation_flag, annotation_size=15, annotation_fmt='%.2f',
                         flag_cbar=flag_cbar, cmap=cmap,
                         fig_size=(18, 12))

    ##
    plot_is_geodataframe(gpd_annual_is,
                         column_name='is_area_increase_pct_1988_2020',
                         title=f'IS increase percentage from 1988 to 2020',
                         vmin=10, vmax=50,
                         annotation_flag=annotation_flag, annotation_size=16, annotation_fmt='%d',
                         flag_cbar=flag_cbar,
                         cmap=plt.get_cmap('YlOrRd'),
                         fig_size=(18, 12))

    ##
    plot_is_geodataframe(gpd_annual_is,
                         column_name='is_reduction_rate_fc',
                         title=f'Reduction of IS increase rate due to the 2008 financial crisis',
                         vmin=-70, vmax= 10,
                         annotation_flag=annotation_flag, annotation_size=16, annotation_fmt='%d',
                         flag_cbar=flag_cbar,
                         cmap=plt.get_cmap('GnBu_r'),
                         fig_size=(18, 12))

    plot_is_geodataframe(gpd_annual_is,
                         column_name='is_recovery_rate_fc',
                         title=f'Recovery of IS increase rate after the 2008 financial crisis',
                         vmin=20, vmax=80,
                         annotation_flag=annotation_flag, annotation_size=16, annotation_fmt='%d',
                         flag_cbar=flag_cbar,
                         cmap=plt.get_cmap('YlOrRd'),
                         fig_size=(18, 12))








