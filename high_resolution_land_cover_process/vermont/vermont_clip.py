"""
    clip
"""

import time
import numpy as np
import shapely
import fiona
import geopandas as gpd
import os
from os.path import join, exists
import sys
from osgeo import gdal, gdal_array, gdalconst
import pandas as pd
import glob
import rasterio
import logging
from shapely.geometry import Polygon

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from high_resolution_land_cover_process.ccap_process.ccap_project_clip_high_res_land_cover import get_img_extent
from high_resolution_land_cover_process.urban_watch.urban_watch_proj_clip import urban_watch_clip_high_resolution_land_cover_to_ard_tile
from high_resolution_land_cover_process.chesapeake_bay_process.cb_2022_clip import cb_clip_high_resolution_land_cover_to_ard_tile

def get_projection_clip_logger(path_vermont):

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger_projection = logging.getLogger('logger_projection_clip')
    logger_projection.setLevel(logging.INFO)

    file_handler_projection = logging.FileHandler(join(path_vermont, 'logger_clip_to_ard.log'))
    file_handler_projection.setLevel(logging.INFO)
    file_handler_projection.setFormatter(formatter)
    logger_projection.addHandler(file_handler_projection)

    return logger_projection


# def main():
if __name__ == '__main__':

    path_vermont = join(rootpath, 'data', 'high_resolution_land_cover', 'Vermont')

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)
    proj_ard = gpd_ard.crs

    logger_proj_clip = get_projection_clip_logger(path_vermont)

    filename_vermont = join(path_vermont, 'ard_proj_land_cover', f'vermont_2016.tif')

    (min_x, min_y, max_x, max_y) = get_img_extent(filename_vermont)  # get the image extent based on the original projection
    boundary_based_on_ard = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])  # get the boundary

    for j in range(0, len(gpd_ard)):
        if boundary_based_on_ard.intersects(gpd_ard.loc[j, 'geometry']):
            tile_name = 'h{:02d}v{:02d}'.format(gpd_ard.loc[j, 'h'], gpd_ard.loc[j, 'v'])
            bounds_intersect_ard = gpd_ard.loc[j, 'geometry'].bounds

            print(tile_name)

            filename_output_clip = join(path_vermont, 'clip_ard_high_resolution',  f'vermont_2016_high_resolution_lc_{tile_name}.tif')

            logger_proj_clip.info(f'{tile_name}, {filename_output_clip}, processing')
            cb_clip_high_resolution_land_cover_to_ard_tile(filename_vermont, filename_output_clip, proj_ard, bounds_intersect_ard)
            logger_proj_clip.info(f'{tile_name}, {filename_output_clip}, {filename_output_clip}, done')
