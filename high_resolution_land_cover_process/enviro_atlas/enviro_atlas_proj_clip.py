"""
    This script is used to project the original land cover from EnviroAtlas to the ARD projection and clip the land cover to ARD tiles
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
from Basic_tools.add_pyramids import save_with_compression


def get_projection_clip_logger(path_2022_cb):

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger_projection = logging.getLogger('logger_projection_clip')
    logger_projection.setLevel(logging.INFO)

    file_handler_projection = logging.FileHandler(join(path_2022_cb, 'logger_projection_clip.log'))
    file_handler_projection.setLevel(logging.INFO)
    file_handler_projection.setFormatter(formatter)
    logger_projection.addHandler(file_handler_projection)

    return logger_projection


def enviro_atlas_proj_ori_land_cover_to_ard_projection(filename_tif, filename_proj_ard_land_cover, proj_ard):
    obj_merged_land_cover = gdal.Open(filename_tif, gdalconst.GA_ReadOnly)

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
                              dstNodata=128,
                              creationOptions=['COMPRESS=LZW']
                              )
    dst = gdal.Warp(destNameOrDestDS=filename_proj_ard_land_cover, srcDSOrSrcDSTab=filename_tif, options=params)
    dst = None
    del dst


def enviro_atlas_get_filename(path_enviro_atlas, folder_name):
    """
    Get the filename of the land cover tif file from the EnviroAtlas dataset

    :param path_enviro_atlas:
    :param folder_name:
    :return:
    """

    filename_tif = join(path_enviro_atlas, 'original_land_cover', folder_name, f'{folder_name[0:-4]}.tif')

    # Some tif files are in a subfolder with the same name as the folder
    if not exists(filename_tif):
        filename_tif = join(path_enviro_atlas, 'original_land_cover', folder_name, f'{folder_name[0:-4]}', f'{folder_name[0:-4]}.tif')

    if not exists(filename_tif):
        filename_tif = join(path_enviro_atlas, 'original_land_cover', folder_name, folder_name, f'{folder_name[0:-4]}.tif')

    return filename_tif


def add_pyramids_color_table_in_enviro_atlas_land_cover(filename_convert_land_cover, df_enviro_atlas_land_cover, list_overview=None):
    """
    add pyramids and color table in the converted enviro_atlas land cover

    :param filename_convert_land_cover:
    :param df_enviro_atlas_land_cover:
    :param list_overview:
    :return:
    """

    dataset = gdal.Open(filename_convert_land_cover, gdal.GA_Update)

    # Generate overviews/pyramids
    # The list [2, 4, 8, 16, 32] defines the downsampling factors for the overviews
    # if list_overview is None:
    #     list_overview = [2, 4, 8, 16, 32, 64]

    # dataset.BuildOverviews(overviewlist=list_overview)

    # Get the first band of the image
    band = dataset.GetRasterBand(1)

    # Create a new color table
    color_table = gdal.ColorTable()

    # Set the color for each value in the color table
    for i in range(0, len(df_enviro_atlas_land_cover)):

        lc_rgb = np.array([df_enviro_atlas_land_cover.loc[i, 'R'], df_enviro_atlas_land_cover.loc[i, 'G'], df_enviro_atlas_land_cover.loc[i, 'B'], 255])
        lc_id = int(df_enviro_atlas_land_cover.loc[i, 'land_cover_id'])

        color_table.SetColorEntry(lc_id, tuple(lc_rgb))

    # Assign the color table to the band
    band.SetRasterColorTable(color_table)

    # Save the changes and close the dataset
    dataset = None


# def main():
if __name__ == '__main__':

    path_enviro_atlas = join(rootpath, 'data', 'high_resolution_land_cover', 'EnviroAtlas')

    df_enviro_atlas = pd.read_excel(join(path_enviro_atlas, 'EnviroAtlas_data_table.xlsx'), sheet_name='dataset_info')
    df_enviro_atlas_land_cover = pd.read_excel(join(path_enviro_atlas, 'EnviroAtlas_data_table.xlsx'), sheet_name='land_cover_info')

    logger_proj_clip = get_projection_clip_logger(path_enviro_atlas)

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)
    proj_ard = gpd_ard.crs

    # for i in range(25, len(df_enviro_atlas)):
    for i in range(24, 25):

        folder_name = df_enviro_atlas.loc[i, 'city_name']
        year = df_enviro_atlas.loc[i, 'year']

        filename_tif = enviro_atlas_get_filename(path_enviro_atlas, folder_name)

        print(i, folder_name, year)

        filename_proj_ard_land_cover = join(path_enviro_atlas, 'ard_proj_land_cover', folder_name, f'{folder_name}_{year}.tif')

        if folder_name == 'SDCA_MULC.tif':
            print('single processing for SDCA_MULC.tif')
            # SDCA_MULC.tif has some projection issue for addressing, so the projection is done in QGIS
            filename_proj_ard_land_cover_qgis = join(path_enviro_atlas, 'ard_proj_land_cover', folder_name, f'SDCA_MULC.tif_2014_qgis.tif')
            save_with_compression(filename_proj_ard_land_cover_qgis, filename_proj_ard_land_cover, compression='LZW')
        else:

            logger_proj_clip.info(f'{folder_name}, {filename_tif}, processing')
            enviro_atlas_proj_ori_land_cover_to_ard_projection(filename_tif, filename_proj_ard_land_cover, proj_ard)
            logger_proj_clip.info(f'{folder_name}, {filename_tif}, done')

        add_pyramids_color_table_in_enviro_atlas_land_cover(filename_proj_ard_land_cover, df_enviro_atlas_land_cover, list_overview=None)

        # clip the land cover to ARD tiles
        (min_x, min_y, max_x, max_y) = get_img_extent(filename_proj_ard_land_cover)  # get the image extent based on the original projection
        boundary_based_on_ard = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])  # get the boundary

        for j in range(0, len(gpd_ard)):
            if boundary_based_on_ard.intersects(gpd_ard.loc[j, 'geometry']):
                tile_name = 'h{:02d}v{:02d}'.format(gpd_ard.loc[j, 'h'], gpd_ard.loc[j, 'v'])
                bounds_intersect_ard = gpd_ard.loc[j, 'geometry'].bounds

                print(tile_name)

                filename_output_clip = join(path_enviro_atlas, 'clip_ard_high_resolution', folder_name, f'{folder_name}_{year}_high_resolution_lc_{tile_name}.tif')

                # add_pyramids_color_table_in_enviro_atlas_land_cover(filename_output_clip, df_enviro_atlas_land_cover, list_overview=None)

                logger_proj_clip.info(f'{folder_name}, {year}, {filename_output_clip}, processing')
                urban_watch_clip_high_resolution_land_cover_to_ard_tile(filename_proj_ard_land_cover, filename_output_clip, proj_ard, bounds_intersect_ard)
                logger_proj_clip.info(f'{folder_name}, {year}, {filename_output_clip}, done')

        ##





















