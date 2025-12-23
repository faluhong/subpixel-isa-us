"""
    clip the merged Chesapeake Bay ISP image with the Landsat ARD tile using the gdal
"""

import os
import sys
import fiona
import pandas as pd
import rasterio
import rasterio.mask
import glob
from osgeo import gdal, ogr, gdal_array, gdalconst
from os.path import join, exists
import numpy as np
import geopandas as gpd
import re

from high_resolution_land_cover_process.mosaic_isp_to_conus import add_pyramids_color_in_nlcd_isp_tif

pwd = os.getcwd()
rootpath = os.path.abspath(join(os.getcwd(), "../../.."))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import HistPlot_ISP, FP, FP_ISP


# def main():
if __name__ =='__main__':

    rootpath_nlcd = r'K:\Data\NLCD\NLCD_impervious_2021_release_all_files_20230630'
    list_available_nlcd_filename = glob.glob(join(rootpath_nlcd, 'nlcd_*_impervious_l48_*.img'))

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)

    proj_ard = gpd_ard.crs.to_wkt()

    ##

    array_tiles = ['h027v008', 'h028v008', 'h027v009']
    array_year_cb = np.array([2013, 2014, 2017, 2018])  # The year of the Chesapeake Bay ISP data

    for i_tile in range(0, len(array_tiles)):
    # for i_tile in range(0, 1):

        tile_name = array_tiles[i_tile]
        print(tile_name)

        # Regular expression to find all integers in the string
        matches = re.findall(r'\d+', tile_name)

        # Convert the found matches to integers
        integers = [int(match) for match in matches]

        h_index = integers[0]
        v_index = integers[1]

        mask_match = (gpd_ard['h'] == h_index) & (gpd_ard['v'] == v_index)
        bounds_intersect_ard = gpd_ard['geometry'][mask_match].iloc[0].bounds

        for i_year in range(0, len(array_year_cb)):
        # for i_year in range(0, 1):

            year = array_year_cb[i_year]
            print(tile_name, year)

            filename_merged_cb_isp = join(rootpath, 'data' , 'ISP_from_high_res_lc', 'merged', f'2022_Chesapeake_Bay_{year}_merged_conus_isp.tif')
            print(filename_merged_cb_isp)

            filename_output = join(rootpath, 'data', 'ISP_from_high_res_lc', '2022_Chesapeake_Bay', tile_name, f'cb_with_tree_isp_{tile_name}_{year}.tif')
            if not os.path.exists(os.path.dirname(filename_output)):
                os.makedirs(os.path.dirname(filename_output), exist_ok=True)

            obj_ori_isp = gdal.Open(filename_merged_cb_isp, gdalconst.GA_ReadOnly)
            proj_ori_isp = obj_ori_isp.GetProjection()

            RES = 30
            params = gdal.WarpOptions(format='GTiff',
                                      outputType=gdalconst.GDT_Byte,
                                      srcSRS=proj_ori_isp,
                                      dstSRS=proj_ard,
                                      outputBounds=bounds_intersect_ard,
                                      xRes=RES,
                                      yRes=RES,
                                      resampleAlg=gdal.GRIORA_NearestNeighbour,
                                      creationOptions=['COMPRESS=LZW']
                                      )
            dst = gdal.Warp(destNameOrDestDS=filename_output, srcDSOrSrcDSTab=filename_merged_cb_isp, options=params)
            dst = None
            del dst





