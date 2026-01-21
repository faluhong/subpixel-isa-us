"""
    extract the CONUS ISP centroid movement
    
    128 GB is not enough, try 256 GB memory
    
    It needs 256 GB memory to run this script. 
    
    The running time for 1988 to 2020 is about 3 hour.
"""

import os
from os.path import join, exists
import sys
import pandas as pd
from osgeo import gdal_array
import numpy as np
import geopandas as gpd
from pyproj import CRS, Transformer

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.conus_isa_centroid.is_centroid_extraction import (get_map_year, get_row_col_id_from_lat_long)


def get_us_proj_information():
    """
        get the projection information for the US land boundary

        :return: proj, geo_transform
    """

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)

    proj =  gpd_ard.crs.to_wkt()

    geo_transform = (gpd_ard.total_bounds[0], 30, 0, gpd_ard.total_bounds[3], 0, -30)

    return (proj, geo_transform)


def get_conus_centroid_loc(img_state_isp_stack):
    """
        Calculate the CONUS centroid of the impervious surface area (ISP) stack

        :param img_state_isp_stack:
        :return:
    """

    img_state_isp_stack = img_state_isp_stack.astype(np.float32)  # ensure the data type is float32
    total = np.nansum(img_state_isp_stack)

    if total == 0:
        print("Warning: Total weight is zero, cannot compute centroid.")

    # Create coordinate grids. It takes huge memory, so use it with caution.
    y, x = np.indices(img_state_isp_stack.shape)

    # Weighted average of coordinates
    cy = np.nansum(y * img_state_isp_stack) / total
    cx = np.nansum(x * img_state_isp_stack) / total

    return (cy, cx)


def get_conus_centroid_loc_manually(img_state_isp_stack):
    """
        Calculate the CONUS centroid of the impervious surface area (ISP) stack manually to save memory

        :param img_state_isp_stack:
        :return:
    """

    total = np.nansum(img_state_isp_stack)

    if total == 0:
        print("Warning: Total weight is zero, cannot compute centroid.")

    # manually get the coordinates grid
    array_y = np.zeros(img_state_isp_stack.shape, dtype=np.float32)
    for i_row in range(np.shape(array_y)[0]):
        array_y[i_row] = i_row

    # calculate the weighted centroid in y-axis
    cy = np.nansum(array_y * img_state_isp_stack) / total

    del array_y  # free memory

    # calculate the weighted centroid in x-axis
    array_x = np.zeros(img_state_isp_stack.shape, dtype=np.float32)
    for i_col in range(np.shape(array_x)[1]):
        array_x[:, i_col] = i_col

    cx = np.nansum(array_x * img_state_isp_stack) / total
    del array_x  # free memory

    return (cy, cx)


def generate_geodataframe_conus_centroid_info(cx, cy, year):
    """
        Generate a GeoDataFrame containing the CONUS centroid information for a given year.

        :param cx:
        :param cy:
        :param year:
        :return:
    """

    # get US projection information
    (proj_ard, geo_transform) = get_us_proj_information()

    proj_wgs84 = CRS("WGS84")
    transformer = Transformer.from_proj(proj_ard, proj_wgs84)

    df_centroid = pd.DataFrame(columns=['year', 'latitude', 'longitude', 'centroid_tile', 'centroid_row', 'centroid_col'],
                               index=np.arange(0, 1))

    # get the latitude and longitude of the centroid, then get the centroid tile, row and column ID, store in the dataframe

    x_centroid = geo_transform[0] + cx * geo_transform[1]
    y_centroid = geo_transform[3] + cy * geo_transform[5]

    latitude_centroid, longitude_centroid = transformer.transform(x_centroid, y_centroid)

    (centroid_tile, centroid_row, centroid_col) = get_row_col_id_from_lat_long(latitude_centroid,
                                                                               longitude_centroid, )

    df_centroid.loc[0, 'year'] = year
    df_centroid.loc[0, 'latitude'] = latitude_centroid
    df_centroid.loc[0, 'longitude'] = longitude_centroid

    df_centroid.loc[0, 'centroid_tile'] = centroid_tile
    df_centroid.loc[0, 'centroid_row'] = centroid_row
    df_centroid.loc[0, 'centroid_col'] = centroid_col

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

    return gdf_centroid


def main():
# if __name__ == '__main__':

    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    filename_prefix = 'unet_regressor_round_masked_post_processing'

    output_folder_merged_conus_isp = 'merge_conus_isp_post_processing_binary_is_ndvi015_sm'  # folder to store the merged conus isp
    filename_output = 'conus_isp_post_processing_binary_is_ndvi015_sm'  # filename prefix of the merged CONUS ISP

    array_year = get_map_year(data_flag)

    gdf_centroid_all = gpd.GeoDataFrame()

    for year in array_year:
        print(f'Processing year: {year}')

        filename_conus_is_pct = join(rootpath, 'results', 'conus_isp', output_folder_merged_conus_isp,
                                     f'{filename_output}_{year}.tif')

        assert exists(filename_conus_is_pct), f'{filename_conus_is_pct} does not exist'

        img_conus_isp = gdal_array.LoadFile(filename_conus_is_pct)
        print(img_conus_isp.dtype)
        img_conus_isp[img_conus_isp == 255] = 0  # set the no data value to NaN

        (cy, cx) = get_conus_centroid_loc_manually(img_conus_isp)
        print(f'Centroid coordinates (cy, cx): ({cy}, {cx})')

        del img_conus_isp  # free memory

        gdf_centroid = generate_geodataframe_conus_centroid_info(cx, cy, year)

        # append the GeoDataFrame to the gdf_centroid_all
        gdf_centroid_all = pd.concat([gdf_centroid_all, gdf_centroid], ignore_index=True)

    ## output the centroid information to a GeoPackage file
    if data_flag == 'conus_isp':
        output_path = join(rootpath, 'results', 'is_centroid', f'conus_level', data_flag, f'{isp_folder}')

    elif data_flag == 'annual_nlcd':
        output_path = join(rootpath, 'results', 'isp_change_stats', f'conus_level', data_flag)
    else:
        raise ValueError('The data_flag is not correct')

    if not exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_filename = join(output_path, f'conus_centroid.gpkg')

    gdf_centroid_all.to_file(output_filename, driver='GPKG')


if __name__ == '__main__':
    main()


