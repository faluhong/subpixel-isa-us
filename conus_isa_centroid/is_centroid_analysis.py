"""
    calculate the IS centroid based on ISP pixel values
    
    For MSA-level, it is recommended to use 96G memory for running the code. It takes about 2 hours when using 10 cores for running.
    
    For county-level, it is recommended to use 96G memory for running the code. It takes about X hours when using 15 cores for running.
    
    For stata-level, it is recommended to run for different year, so new script is needed
    
    Note 2026-06-03: The previous code requires a lot of memory, so it is modified to run for each year independently.
                     The current code is more like a utility function code
"""

import os
from os.path import join, exists
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import matplotlib
import numpy as np
import seaborn as sns
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable     # noinspection PyUnresolvedReferences
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from pyproj import CRS, Transformer
import geopandas as gpd
import click

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP

from conus_isp_analysis.state_is_area_cal import (load_state_isp_stack, extract_whole_region_mask)

from deep_learning_isp.utils_deep_learning import get_proj_info
from analysis.utils_isp_time_series_analysis import (get_row_col_id_from_lat_long)

from evaluation.utils_plot_isp import isp_colorbar


def get_map_year(data_flag):
    """
        get the year to extract the centroid based on the data_flag

        :param data_flag: 'conus_isp' or 'annual_nlcd'
        :return:
    """

    if data_flag == 'conus_isp':
        array_target_year = np.arange(1988, 2021, 1)
    elif data_flag == 'annual_nlcd':
        array_target_year = np.arange(1985, 2024, 1)
    else:
        raise ValueError(f'Unknown data_flag: {data_flag}')

    return array_target_year


def get_target_basic_info(modify_target):
    """
        Get the basic information of the target region based on the modify_target.

        :param modify_target: 'state', 'msa' or 'county'
        :return:
    """

    if (modify_target == 'state') | (modify_target == 'county'):

        filename_output = join(rootpath, 'data', 'shapefile', 'CONUS_boundary',
                               f'tl_2023_us_{modify_target}', f'conus_{modify_target}_basic_info.csv')
        df_conus_target_basic_info = pd.read_csv(filename_output)
    else:
        path_msa_2015 = join(rootpath, 'data', 'urban_pulse', 'shapefile', 'cb_2015_us_cbsa_500k')

        filename_output = join(path_msa_2015, 'cb_2015_us_cbsa_500k_ard_conus.csv')
        df_conus_target_basic_info = pd.read_csv(filename_output)

    return df_conus_target_basic_info


def get_weighted_centroid(image):
    """
    Calculate the centroid of the image based on pixel values.

    Parameters:
    - image: 2D NumPy array (e.g., grayscale image or mask)

    Returns:
    - (cy, cx): tuple of floats representing the weighted centroid (row, column)
    """

    # image = np.asarray(image, dtype=float)
    total = np.nansum(image)

    if total == 0:
        print("Warning: Total weight is zero, cannot compute centroid.")
        return None  # Avoid division by zero; no weights present

    # Create coordinate grids
    y, x = np.indices(image.shape)

    # Weighted average of coordinates
    cy = np.nansum(y * image) / total
    cx = np.nansum(x * image) / total

    return (cy, cx)


def load_isp_admin_boundary_stack(data_flag, isp_folder, filename_prefix,
                                  df_conus_target_basic_info, i_target,
                                  modify_target,
                                  array_target_year=None):
    """
        load the impervious surface area (ISP) stack for a specific adminstration boundary in the CONUS ISP dataset.

        :param data_flag: 'conus_isp' or 'annual_nlcd'
        :param isp_folder:
        :param filename_prefix:
        :param i_target: index in the CONUS target basic info dataframe, 10 is for Connecticut at state level, 1345 is for Connecticut at county level
        :param modify_target: 'state', 'msa' or 'county', default is 'county'

        # isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
        # filename_prefix = 'unet_regressor_round_masked_post_processing'

        :return:
    """

    print(modify_target)

    state_name = df_conus_target_basic_info['NAME'].values[i_target]

    if array_target_year is None:
        array_target_year = get_map_year(data_flag)

    mask_whole_state = extract_whole_region_mask(df_conus_target_basic_info, i_target,
                                                 nrow=5000, ncol=5000,
                                                 modify_target=modify_target)

    print(state_name, array_target_year)

    # load the ISP stack for calculation
    img_state_isp_stack = load_state_isp_stack(df_conus_target_basic_info,
                                               i_state=i_target,
                                               array_year=array_target_year,
                                               data_flag=data_flag,
                                               isp_folder=isp_folder,
                                               filename_prefix=filename_prefix,
                                               nlcd_folder='NLCD_annual',
                                               nlcd_filter_ocean_flag=False,
                                               rootpath_conus_isp=None,
                                               rootpath_nlcd_directory=None, )

    # set the out-state pixels to 255, i.e., nodata value
    img_state_isp_stack[:, mask_whole_state == False] = 255

    img_state_isp_stack = img_state_isp_stack.astype(float)
    img_state_isp_stack[img_state_isp_stack == 255] = np.nan

    return (img_state_isp_stack)


def get_annual_centroid_loc(img_state_isp_stack):
    """
        Calculate the annual centroid of the impervious surface area (ISP) stack

        :param img_state_isp_stack:
        :return:
    """

    len_year = np.shape(img_state_isp_stack)[0]

    array_cy = np.zeros(len_year, dtype=float)
    array_cx = np.zeros(len_year, dtype=float)

    for i_year in range(0, len_year):
    # for i_year in range(0, 1):

        # year = array_year[i_year]
        # print(f'Processing year: {year}')

        img_isp_single_year = img_state_isp_stack[i_year, :, :]

        img_isp_single_year = img_isp_single_year.astype(float)
        img_isp_single_year[img_isp_single_year == 255] = np.nan

        (cy, cx) = get_weighted_centroid(img_isp_single_year)

        array_cy[i_year] = cy
        array_cx[i_year] = cx

        print(f'{i_year}, Centroid: (cy, cx) = ({cy:.2f}, {cx:.2f})')

    return (array_cy, array_cx)


def generate_geodataframe_centroid_info(array_cy, array_cx, df_conus_state_basic_info, i_target, array_year,):
    """
        generate the GeoDataFrame to store the centroid information for the impervious surface centroid movement

        :param array_cy:
        :param array_cx:
        :param df_conus_state_basic_info:
        :param i_target:
        :param array_year:
        :return:
    """

    # define the dataframe to store the centroid information
    df_centroid = pd.DataFrame(columns=list(df_conus_state_basic_info.columns) + ['year', 'latitude', 'longitude',
                                                                                  'centroid_tile', 'centroid_row', 'centroid_col'],
                               index=np.arange(0, len(array_year)))

    # get the minimum and maximum tile h and v ID
    h_min = df_conus_state_basic_info['h_min'].values[i_target]
    v_min = df_conus_state_basic_info['v_min'].values[i_target]

    # get the projection information for the tile
    proj_ard, geo_transform = get_proj_info(tile_name=f'h{h_min:03d}v{v_min:03d}', )

    proj_wgs84 = CRS("WGS84")
    transformer = Transformer.from_proj(proj_ard, proj_wgs84)

    # get the latitude and longitude of the centroid, then get the centroid tile, row and column ID, store in the dataframe
    for i_centroid in range(len(array_cy)):
        row_loc = array_cy[i_centroid]
        col_loc = array_cx[i_centroid]

        x_centroid = geo_transform[0] + col_loc * geo_transform[1]
        y_centroid = geo_transform[3] + row_loc * geo_transform[5]

        latitude_centroid, longitude_centroid = transformer.transform(x_centroid, y_centroid)

        (centroid_tile, centroid_row, centroid_col) = get_row_col_id_from_lat_long(latitude_centroid,
                                                                                   longitude_centroid, )

        df_centroid.loc[i_centroid, df_conus_state_basic_info.columns] = df_conus_state_basic_info.iloc[i_target, :].values
        df_centroid.loc[i_centroid, 'year'] = array_year[i_centroid]
        df_centroid.loc[i_centroid, 'latitude'] = latitude_centroid
        df_centroid.loc[i_centroid, 'longitude'] = longitude_centroid

        df_centroid.loc[i_centroid, 'centroid_tile'] = centroid_tile
        df_centroid.loc[i_centroid, 'centroid_row'] = centroid_row
        df_centroid.loc[i_centroid, 'centroid_col'] = centroid_col

    # convert the dataframe to geopandas dataframe
    geometry = gpd.points_from_xy(df_centroid['longitude'], df_centroid['latitude'])

    gdf_centroid = gpd.GeoDataFrame(df_centroid,
                                    geometry=geometry,
                                    crs="EPSG:4326")
    
    for col in gdf_centroid.columns:
        if gdf_centroid[col].dtype == 'int64':
            gdf_centroid[col] = gdf_centroid[col].astype(int)
        elif gdf_centroid[col].dtype == 'float64':
            gdf_centroid[col] = gdf_centroid[col].astype(float)
        elif gdf_centroid[col].dtype == 'object':
            gdf_centroid[col] = gdf_centroid[col].astype(str)

    return (gdf_centroid)


def get_centroid_gpkg_output_filename(modify_target, df_conus_target_basic_info, i_target, data_flag, isp_folder, ):
    """
        Get the output filename for the centroid GeoPackage file.

        :param modify_target: 'state', 'msa' or 'county'
        :param df_conus_target_basic_info:
        :param i_target:
        :param data_flag:
        :param isp_folder:
        :param gdf_centroid: the GeoDataFrame containing the centroid information
        :param flag_output: whether to output the GeoPackage file, default is True
        :param flag_return_filename: whether to return the output filename, default is False

        :return:
    """

    if modify_target == 'state':
        state_short_name = df_conus_target_basic_info['STUSPS'].values[i_target]
    elif modify_target == 'msa':
        msa_name = df_conus_target_basic_info['NAME'].values[i_target]
        # replace the '/' with '_' to avoid the error in the file name, such as Louisville/Jefferson County, KY-IN
        msa_name = msa_name.replace(r'/', '_')
    elif modify_target == 'county':
        state_short_name = df_conus_target_basic_info['STATE_STUSPS'].values[i_target]
        state_name = df_conus_target_basic_info['STATE_NAME'].values[i_target]
        # use 'NAMELSAD' instead of 'NAME' to avoid the same county short name in the same state
        county_name = df_conus_target_basic_info['NAMELSAD'].values[i_target]
    else:
        raise ValueError('The modify_target is not correct')

    # get the output path
    if (modify_target == 'state') | (modify_target == 'msa'):
        if data_flag == 'conus_isp':
            output_path = join(rootpath, 'results', 'is_centroid', f'{modify_target}_level', data_flag, f'{isp_folder}')

        elif data_flag == 'annual_nlcd':
            output_path = join(rootpath, 'results', 'is_centroid', f'{modify_target}_level', data_flag)
        else:
            raise ValueError('The data_flag is not correct')
    elif (modify_target == 'county'):
        if data_flag == 'conus_isp':
            output_path = join(rootpath, 'results', 'is_centroid', f'{modify_target}_level', data_flag, f'{isp_folder}', f'{state_name}')

        elif data_flag == 'annual_nlcd':
            output_path = join(rootpath, 'results', 'is_centroid', f'{modify_target}_level', data_flag, f'{state_name}')
        else:
            raise ValueError('The data_flag is not correct')
    else:
        raise ValueError('The data_flag is not correct')

    if not exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if modify_target == 'state':
        output_filename = join(output_path, f'{state_short_name}_centroid.gpkg')
    elif modify_target == 'msa':
        output_filename = join(output_path, f'{msa_name}_centroid.gpkg')
    elif modify_target == 'county':
        output_filename = join(output_path, f'{state_short_name}_{county_name}_centroid.gpkg')
    else:
        raise ValueError('The modify_target is not correct')

    return output_filename


def plot_isp_centroid_movement(img, array_cx, array_cy, array_year,
                               vmin=0, vmax=100,
                               ax_plot=None,
                               figsize=(14, 12),
                               ticks_flag=True,
                               title=None,
                               flag_cbar=True,
                               ):
    """
        plot the impervious surface area (ISP) centroid movement over the years

        The ISP image is plotted as a background, and the centroid path is plotted as a scatter plot.

        :return:
    """

    sns.set_style("white")

    # change the value outside the range to nan
    img = img.astype(float)
    img[img < vmin] = np.nan
    img[img > vmax] = np.nan

    cmap_isp = isp_colorbar()
    norm = mcolors.Normalize(vmin=0, vmax=100)

    # plot the image
    title_label_size = 25
    axis_tick_label_size = 18
    axis_label_size = 24
    cbar_tick_label_size = 24
    tick_length = 6
    axes_line_width = 1.5

    if ax_plot is None:
        fig, ax_plot = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    matplotlib.rcParams['axes.linewidth'] = axes_line_width
    for i in ax_plot.spines.values():
        i.set_linewidth(axes_line_width)

    # ISP plot
    im_isp = ax_plot.imshow(img, cmap=cmap_isp, norm=norm, interpolation='nearest')

    # centroid plot
    viridis = plt.get_cmap('viridis')

    # Truncate it: second half (0.5 to 1.0)
    # half_viridis = mcolors.LinearSegmentedColormap.from_list(
    #     'half_viridis', viridis(np.linspace(0.5, 1.0, 256)))

    im_centroid = ax_plot.scatter(array_cx, array_cy, s=100, c=array_year, cmap=viridis, linewidth=1.0, label='Centroid Path')

    ax_plot.tick_params('x', labelsize=axis_tick_label_size, direction='out', length=tick_length,
                        width=axes_line_width, bottom=ticks_flag, labelbottom=ticks_flag, which='major')
    ax_plot.tick_params('y', labelsize=axis_tick_label_size, direction='out', length=tick_length,
                        width=axes_line_width, left=ticks_flag, labelleft=ticks_flag, which='major')

    if flag_cbar:
        # define the colorbar
        divider = make_axes_locatable(ax_plot)

        # color bar for the year
        cax = divider.append_axes("right", size="6%", pad=0.3)

        cb = plt.colorbar(im_centroid, cax=cax, cmap=viridis)
        cb.ax.tick_params(labelsize=cbar_tick_label_size)
        cb.ax.set_ylabel('Year', size=axis_label_size)

        cb.formatter = FormatStrFormatter('%d')  # Use '%d' for integer formatting

        cb.set_ticks(array_year)

        cb.update_ticks()

        # color bar for the ISP percentage
        cax = divider.append_axes("right", size="6%", pad=0.3 + 1.0)

        cb = plt.colorbar(im_isp, cax=cax, cmap=cmap_isp, norm=norm, drawedges=True, ticks=np.arange(0, 101, 10))

        # split the colorbar into sets
        cb.outline.set_edgecolor('w')
        cb.dividers.set_edgecolor('w')
        cb.dividers.set_linewidth(0.5)

        cb.ax.tick_params(labelsize=cbar_tick_label_size, length=tick_length)
        cb.ax.set_ylabel('Impervious Surface Percentage (%)', size=axis_label_size)

    ax_plot.set_title(title, fontsize=title_label_size)

    plt.tight_layout()
    plt.show()


# @click.command()
# @click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
# @click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
# @click.option('--modify_target', type=str, default='msa', help='the target to modify, e.g., msa, county or state')
# def main(rank, n_cores, modify_target):
if __name__ == '__main__':

    rank = 1
    n_cores = 100000
    modify_target = 'state' # 'msa', 'county' or 'state'

    # 1345 for Connecticut county in the CONUS basic info dataframe, 10 for Connecticut state, 596 for Connecticut Hartford MSA
    # i_target = 596

    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    filename_prefix = 'unet_regressor_round_masked_post_processing'

    array_year = get_map_year(data_flag)

    df_conus_target_basic_info = get_target_basic_info(modify_target=modify_target)

    each_core_block = int(np.ceil(len(df_conus_target_basic_info) / n_cores))
    for i in range(0, each_core_block):
        new_rank = rank - 1 + i * n_cores
        # means that all folder has been processed
        if new_rank > len(df_conus_target_basic_info) - 1:
            print(f'{new_rank} this is the last running task')
        else:

            # load CT ISP stack, i_target = 10
            # load the CT Northwest Hills county ISP stack, i_target = 1345
            # load the CT Hartford MSA ISP stack, i_target = 596
            
            gdf_centroid_all = gpd.GeoDataFrame()

            for year in array_year:
                print(f'Processing year: {year}')

                (img_state_isp_stack) = load_isp_admin_boundary_stack(data_flag=data_flag,
                                                                      isp_folder=isp_folder,
                                                                      filename_prefix=filename_prefix,
                                                                      df_conus_target_basic_info=df_conus_target_basic_info,
                                                                      i_target=new_rank,
                                                                      modify_target=modify_target,
                                                                      array_target_year=np.array([year]),)

                # get the annual centroid of the ISP stack
                (array_cy, array_cx) = get_annual_centroid_loc(img_state_isp_stack)
                print(array_cy, array_cx)

                # define the GeoDataFrame to store the centroid information
                gdf_centroid = generate_geodataframe_centroid_info(array_cy=array_cy,
                                                                   array_cx=array_cx,
                                                                   df_conus_state_basic_info=df_conus_target_basic_info,
                                                                   i_target=new_rank,
                                                                   array_year=np.array([year]),)

                # append the GeoDataFrame to the gdf_centroid_all
                gdf_centroid_all = pd.concat([gdf_centroid_all, gdf_centroid], ignore_index=True)

            # output the centroid information to a GeoPackage file
            output_filename = get_centroid_gpkg_output_filename(modify_target=modify_target,
                                                                df_conus_target_basic_info=df_conus_target_basic_info,
                                                                i_target=new_rank,
                                                                data_flag=data_flag,
                                                                isp_folder=isp_folder)

            gdf_centroid.to_file(output_filename, driver='GPKG')
            print(f'Centroid information saved to: {output_filename}')

            ##
            # plot_isp_single(img_isp_single_year, title=f'CT_ISP',)
            # row_start = 1500
            # row_end = 7000
            # col_start = 1600
            # col_end = 6800

            # row_start = 0
            # row_end = np.shape(img_state_isp_stack)[1]
            # col_start = 0
            # col_end = np.shape(img_state_isp_stack)[2]

            # plot_isp_centroid_movement(img=img_state_isp_stack[-1, row_start:row_end, col_start:col_end].copy(),
            #                            array_cx=array_cx - col_start,
            #                            array_cy=array_cy - row_start,
            #                            array_year=array_year,
            #                            vmin=0, vmax=100,
            #                            ax_plot=None,
            #                            figsize=(14, 12),
            #                            ticks_flag=True,
            #                            title=None,
            #                            flag_cbar=True,
            #                            )

  
# if __name__ == '__main__':
    # main()






