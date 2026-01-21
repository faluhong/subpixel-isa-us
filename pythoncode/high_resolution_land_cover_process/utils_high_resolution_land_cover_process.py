"""
    utility functions to process the high resolution land cover product
"""

import numpy as np
import os
from os.path import join
import sys
from osgeo import gdal, gdalconst

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.model_training.utils_deep_learning import get_proj_info


def get_img_extent(filename_obj):
    """
        get the image extent
        :param
            obj_ccap: objective of the input .tif image
        :return:
    """

    obj_ccap = gdal.Open(filename_obj)
    gt = obj_ccap.GetGeoTransform()

    # Get the raster size
    width = obj_ccap.RasterXSize
    height = obj_ccap.RasterYSize

    # Calculate the image extent
    min_x = gt[0]
    min_y = gt[3] + width * gt[4] + height * gt[5]
    max_x = gt[0] + width * gt[1] + height * gt[2]
    max_y = gt[3]

    # print("Image extent:")
    # print("Min X:", min_x)
    # print("Min Y:", min_y)
    # print("Max X:", max_x)
    # print("Max Y:", max_y)

    return min_x, min_y, max_x, max_y


def clip_high_resolution_land_cover_to_ard_tile(filename_reproj_ard_land_cover, filename_output_clip, proj_ard, bounds_intersect_ard):
    """
    clip the high resolution land cover to the ARD tile

    :param filename_reproj_ard_land_cover:
    :param filename_output_clip:
    :param proj_ard:
    :param bounds_intersect_ard:
    """

    if not os.path.exists(os.path.dirname(filename_output_clip)):
        os.makedirs(os.path.dirname(filename_output_clip), exist_ok=True)

    obj_ori_land_cover = gdal.Open(filename_reproj_ard_land_cover, gdalconst.GA_ReadOnly)
    proj_ori_land_cover = obj_ori_land_cover.GetProjection()

    RES = 1
    params = gdal.WarpOptions(format='GTiff',
                              outputType=gdalconst.GDT_Byte,
                              srcSRS=proj_ori_land_cover,
                              dstSRS=proj_ard.to_wkt(),
                              outputBounds=bounds_intersect_ard,
                              xRes=RES,
                              yRes=RES,
                              resampleAlg=gdal.GRIORA_NearestNeighbour,
                              dstNodata=0,
                              creationOptions=['COMPRESS=LZW']
                              )
    dst = gdal.Warp(destNameOrDestDS=filename_output_clip, srcDSOrSrcDSTab=filename_reproj_ard_land_cover, options=params)
    dst = None
    del dst


def output_estimated_isp(img_isp, tile_name, output_filename):
    """
    output the estimated ISP

    :param img_isp:
    :param tile_name:
    :param folder_name:
    :param year:
    :return:
    """

    proj_ard, geo_transform = get_proj_info(tile_name)
    ncol_30m, nrow_30m = np.shape(img_isp)[1], np.shape(img_isp)[0]

    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    tif_output = gdal.GetDriverByName('GTiff').Create(output_filename, ncol_30m, nrow_30m, 1, gdalconst.GDT_Float32)
    # tif_output = gdal.GetDriverByName('GTiff').Create(output_filename, ncol_30m, nrow_30m, 1, gdalconst.GDT_Byte, options=['COMPRESS=LZW'])
    tif_output.SetGeoTransform(geo_transform)
    tif_output.SetProjection(proj_ard)

    Band = tif_output.GetRasterBand(1)
    Band.WriteArray(img_isp)

    tif_output = None
    del tif_output





