"""
    project and clip the 2022 Chesapeake Bay land cover data

    Note on 2024-Apr-22
    The Pennsylvania State is too large to finish th projection process. But it looks like we do not need to do the projection process.
    We can directly clip the high resolution land cover to the ARD tile.
"""

import time
import numpy as np
import shapely
import fiona
import geopandas as gpd
import os
from os.path import join, exists
import sys
# os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'
from osgeo import gdal, gdal_array, gdalconst
import pandas as pd
import glob
from pyproj import Proj, CRS, Transformer
from shapely.geometry import Polygon
import rasterio
import logging
from rasterio.merge import merge

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from high_resolution_land_cover_process.ccap_process.ccap_project_clip_high_res_land_cover import get_img_extent
from high_resolution_land_cover_process.urban_watch.urban_watch_proj_clip import urban_watch_clip_high_resolution_land_cover_to_ard_tile


def get_projection_clip_logger(path_2022_cb):

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger_projection = logging.getLogger('logger_projection_clip')
    logger_projection.setLevel(logging.INFO)

    file_handler_projection = logging.FileHandler(join(path_2022_cb, 'logger_projection_clip.log'))
    file_handler_projection.setLevel(logging.INFO)
    file_handler_projection.setFormatter(formatter)
    logger_projection.addHandler(file_handler_projection)

    return logger_projection


def proj_ori_land_cover_to_ard_projection(filename_2022_cb, filename_proj_ard_land_cover, proj_ard):
    obj_merged_land_cover = gdal.Open(filename_2022_cb, gdalconst.GA_ReadOnly)

    if not os.path.exists(os.path.dirname(filename_proj_ard_land_cover)):
        os.makedirs(os.path.dirname(filename_proj_ard_land_cover), exist_ok=True)

    RES = 1
    params = gdal.WarpOptions(format='GTiff',
                              outputType=gdalconst.GDT_Byte,
                              srcSRS=obj_merged_land_cover.GetProjection(),
                              dstSRS=proj_ard.to_wkt(),
                              xRes=RES,
                              yRes=RES,
                              resampleAlg=gdal.GRIORA_NearestNeighbour,
                              dstNodata=0,
                              creationOptions=['COMPRESS=LZW',
                                              'BIGTIFF=YES']
                              )
    dst = gdal.Warp(destNameOrDestDS=filename_proj_ard_land_cover, srcDSOrSrcDSTab=filename_2022_cb, options=params)
    dst = None
    del dst


# def main():
if __name__ == '__main__':

    path_2022_cb = join(rootpath, 'data', 'high_resolution_land_cover', '2022_Chesapeake_Bay')

    df_2022_cb = pd.read_excel(join(path_2022_cb, '2022_Chesapeake_Bay_table.xlsx'), sheet_name='Sheet1')

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)
    proj_ard = gpd_ard.crs

    logger_proj_clip = get_projection_clip_logger(path_2022_cb)

    for i in range(8, len(df_2022_cb)):
    # for i in range(0, 1):
        city_name = df_2022_cb.loc[i, 'city_name']
        folder_name = df_2022_cb.loc[i, 'folder_name']
        year = df_2022_cb.loc[i, 'year']

        filename_2022_cb = join(path_2022_cb, 'original_land_cover', city_name, folder_name, f'{folder_name}.tif')

        print(i, city_name, folder_name, year, os.path.exists(filename_2022_cb))

        filename_proj_ard_land_cover = join(path_2022_cb, 'ard_proj_land_cover', city_name, f'{year}', f'{folder_name}.tif')

        logger_proj_clip.info(f'{city_name} {folder_name}, {filename_2022_cb}, processing')
        proj_ori_land_cover_to_ard_projection(filename_2022_cb, filename_proj_ard_land_cover, proj_ard)
        logger_proj_clip.info(f'{city_name} {folder_name}, {filename_2022_cb}, done')

        ##

        (min_x, min_y, max_x, max_y) = get_img_extent(filename_proj_ard_land_cover)  # get the image extent based on the original projection
        boundary_based_on_ard = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])  # get the boundary

        for j in range(0, len(gpd_ard)):
            if boundary_based_on_ard.intersects(gpd_ard.loc[j, 'geometry']):
                tile_name = 'h{:02d}v{:02d}'.format(gpd_ard.loc[j, 'h'], gpd_ard.loc[j, 'v'])
                bounds_intersect_ard = gpd_ard.loc[j, 'geometry'].bounds

                print(tile_name)

                filename_output_clip = join(path_2022_cb, 'clip_ard_high_resolution', city_name, f'{year}', f'{folder_name}_{year}_high_resolution_lc_{tile_name}.tif')

                logger_proj_clip.info(f'{city_name}, {folder_name}, {year}, {filename_output_clip}, processing')
                urban_watch_clip_high_resolution_land_cover_to_ard_tile(filename_proj_ard_land_cover, filename_output_clip, proj_ard, bounds_intersect_ard)
                logger_proj_clip.info(f'{city_name}, {folder_name}, {year}, {filename_output_clip}, done')


