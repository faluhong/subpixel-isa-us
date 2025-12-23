"""
    This script applies permanent natural surface to the post-processed CONUS ISP images
    (1) exclude the pixels outside the CONUS boundary
    (2) apply the permanent natural surface to the post-processed CONUS ISP images
        If the pixel is consistently classified as natural surface in all the years, then it is considered as permanent natural surface
        Set the ISP of these pixels to 0
"""


import os
from os.path import join, exists
import sys
from osgeo import gdal, gdal_array, gdalconst
import pandas as pd
import glob
import numpy as np
import time

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.add_pyramids import add_pyramids_in_tif
from analysis.conus_annual_mean_isp import (get_us_land_boundary, get_conus_isp_falu)
from high_resolution_land_cover_process.mosaic_isp_to_conus import output_mosaic_isp_img


def permanent_natural_surface_mask(filename_mask='conus_permanent_natural_surface_count', rootpath_project_folder=None):

    if rootpath_project_folder is None:
        rootpath_project_folder = rootpath

    filename_permanent_natural_surface_count = join(rootpath_project_folder, 'data', 'permanent_natural_surface_mask',
                                                    f'{filename_mask}.tif')
    img_permanent_natural_surface_count = gdal_array.LoadFile(filename_permanent_natural_surface_count)
    mask_permanent_natural_surface = (img_permanent_natural_surface_count == 38)

    return mask_permanent_natural_surface


# def main():
if __name__ =='__main__':

    rootpath_project_folder = r'/shared/zhulab/Falu/CSM_project/'  # rootpath # r'/shared/zhulab/Falu/CSM_project/'
    # rootpath_project_folder = rootpath   # rootpath # r'/shared/zhulab/Falu/CSM_project/'

    # get the mask of US land, true for land, false for ocean
    mask_us_land = get_us_land_boundary()

    list_year = np.arange(1985, 2023)

    mask_permanent_natural_surface = permanent_natural_surface_mask(filename_mask='conus_permanent_natural_surface_count_v2',
                                                                    rootpath_project_folder=rootpath)

    ##
    for i_year in range(0, len(list_year)):
    # for i_year in range(len(list_year) - 1, len(list_year)):
        year = list_year[i_year]
        print(year)

        # img_conus_isp = get_conus_isp_falu(year, path_conus_isp_falu=None)
        img_conus_isp = get_conus_isp_falu(year, 
                                           path_conus_isp_falu=rootpath_project_folder, 
                                           merge_conus_isp_folder='merge_conus_isp_post_processing_mean', 
                                           merge_conus_isp_filename='conus_isp_post_processing_mean')

        img_conus_isp[mask_us_land == 0] = 255  # exclude the pixels outside the CONUS boundary
        img_conus_isp[mask_permanent_natural_surface] = 0

        # output the masked CONUS ISP image
        output_filename_merged_isp = join(rootpath_project_folder, 'results', 'conus_isp', 
                                          'merge_conus_isp_post_processing_mean_filter',
                                          f'conus_isp_filter_{year}.tif')

        output_mosaic_isp_img(img_conus_isp, output_filename_merged_isp, nrow=5000, ncol=5000, total_v=22, total_h=33,
                              add_pyramids=True)
