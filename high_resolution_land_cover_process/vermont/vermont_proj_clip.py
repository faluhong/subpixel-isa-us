"""
    project and clip the Vermont 0.5-meter land cover dataset

    It takes pretty long time to process the dataset. Have already done it in QGIS
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
from high_resolution_land_cover_process.urban_watch.urban_watch_proj_clip import (urban_watch_reproj_ori_land_cover_to_ard_projection,
                                                                                  urban_watch_clip_high_resolution_land_cover_to_ard_tile)

# def main():
if __name__ == '__main__':

    path_vermont = join(rootpath, 'data', 'high_resolution_land_cover', 'Vermont')

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)
    proj_ard = gpd_ard.crs

    filename_original_land_cover = join(path_vermont, 'LandLandcov_BaseLC2016', 'LandLandcov_BaseLC2016.tif')
    year = 2016

    filename_proj_ard_land_cover = join(path_vermont, 'ard_proj_land_cover', f'vermont_{year}.tif')
    urban_watch_reproj_ori_land_cover_to_ard_projection(filename_original_land_cover, filename_proj_ard_land_cover, proj_ard)

    ##

    (min_x, min_y, max_x, max_y) = get_img_extent(filename_proj_ard_land_cover)  # get the image extent based on the original projection
    boundary_based_on_ard = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])  # get the boundary

    for j in range(0, len(gpd_ard)):
        if boundary_based_on_ard.intersects(gpd_ard.loc[j, 'geometry']):
            tile_name = 'h{:02d}v{:02d}'.format(gpd_ard.loc[j, 'h'], gpd_ard.loc[j, 'v'])
            bounds_intersect_ard = gpd_ard.loc[j, 'geometry'].bounds

            print(tile_name)

            filename_output_clip = join(path_vermont, 'clip_ard_high_resolution', f'vermont_{year}_high_resolution_lc_{tile_name}.tif')

            urban_watch_clip_high_resolution_land_cover_to_ard_tile(filename_proj_ard_land_cover, filename_output_clip, proj_ard, bounds_intersect_ard)























