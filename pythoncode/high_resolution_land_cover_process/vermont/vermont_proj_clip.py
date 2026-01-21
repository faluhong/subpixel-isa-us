"""
    project and clip the Vermont 0.5-meter land cover dataset

    It takes pretty long time to process the dataset. Suggest to do it in QGIS
"""

import geopandas as gpd
import os
from os.path import join
import sys
from osgeo import gdal, gdalconst
from shapely.geometry import Polygon

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.high_resolution_land_cover_process.utils_high_resolution_land_cover_process import (get_img_extent,
                                                                                                    clip_high_resolution_land_cover_to_ard_tile)


def reproj_ori_land_cover_to_ard_projection(filename_merged_land_cover, filename_reproj_ard_land_cover, proj_ard):
    """
    reproject the original merged land cover image to the ARD projection

    The process is necessary to ensure all the data is kept

    :param filename_merged_land_cover:
    :param filename_reproj_ard_land_cover:
    :param proj_ard:
    """

    obj_merged_land_cover = gdal.Open(filename_merged_land_cover, gdalconst.GA_ReadOnly)

    if not os.path.exists(os.path.dirname(filename_reproj_ard_land_cover)):
        os.makedirs(os.path.dirname(filename_reproj_ard_land_cover), exist_ok=True)

    RES = 1
    params = gdal.WarpOptions(format='GTiff',
                              outputType=gdalconst.GDT_Byte,
                              srcSRS=obj_merged_land_cover.GetProjection(),
                              dstSRS=proj_ard.to_wkt(),
                              xRes=RES,
                              yRes=RES,
                              resampleAlg=gdal.GRIORA_NearestNeighbour,
                              dstNodata=0,
                              creationOptions=['COMPRESS=LZW']
                              )
    dst = gdal.Warp(destNameOrDestDS=filename_reproj_ard_land_cover, srcDSOrSrcDSTab=filename_merged_land_cover, options=params)
    dst = None
    del dst


# def main():
if __name__ == '__main__':

    path_vermont = join(rootpath, 'data', 'high_resolution_land_cover', 'Vermont')

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)
    proj_ard = gpd_ard.crs

    filename_original_land_cover = join(path_vermont, 'LandLandcov_BaseLC2016', 'LandLandcov_BaseLC2016.tif')
    year = 2016

    filename_proj_ard_land_cover = join(path_vermont, 'ard_proj_land_cover', f'vermont_{year}.tif')
    reproj_ori_land_cover_to_ard_projection(filename_original_land_cover, filename_proj_ard_land_cover, proj_ard)

    ##

    (min_x, min_y, max_x, max_y) = get_img_extent(filename_proj_ard_land_cover)  # get the image extent based on the original projection
    boundary_based_on_ard = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])  # get the boundary

    for j in range(0, len(gpd_ard)):
        if boundary_based_on_ard.intersects(gpd_ard.loc[j, 'geometry']):
            tile_name = 'h{:02d}v{:02d}'.format(gpd_ard.loc[j, 'h'], gpd_ard.loc[j, 'v'])
            bounds_intersect_ard = gpd_ard.loc[j, 'geometry'].bounds

            print(tile_name)

            filename_output_clip = join(path_vermont, 'clip_ard_high_resolution', f'vermont_{year}_high_resolution_lc_{tile_name}.tif')

            clip_high_resolution_land_cover_to_ard_tile(filename_proj_ard_land_cover, filename_output_clip, proj_ard, bounds_intersect_ard)








