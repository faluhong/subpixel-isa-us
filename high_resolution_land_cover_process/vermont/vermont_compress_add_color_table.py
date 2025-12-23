"""
    compress the vermont land cover generated from QGIS
    add the pyramids and color table in the land cover tif file
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


def vermont_save_with_compression(input_path, output_path, compression='LZW'):
    """
        Save the input raster file with compression
    """

    # Open the input raster file
    input_ds = gdal.Open(input_path)
    if input_ds is None:
        print(f"Could not open {input_path}")
        return

    # Get driver from the input raster
    driver = input_ds.GetDriver()

    # Create output raster dataset with compression
    output_ds = driver.CreateCopy(output_path, input_ds, options=["COMPRESS=" + compression])

    # Close datasets
    input_ds = None
    output_ds = None


def vermont_add_pyramids_color_in_lc_tif(filename_compress, df_land_cover_table, list_overview=None):
    """
        add pyramids and color table in the vermont land cover tif file
    """

    dataset = gdal.Open(filename_compress, gdal.GA_Update)

    # Generate overviews/pyramids
    # The list [2, 4, 8, 16, 32] defines the downsampling factors for the overviews

    if list_overview is None:
        list_overview = [2, 4, 8, 16, 32, 64]

    dataset.BuildOverviews(overviewlist=list_overview)

    # Get the first band of the image
    band = dataset.GetRasterBand(1)

    # Create a new color table
    color_table = gdal.ColorTable()

    # Set the color for each value in the color table
    for i in range(0, len(df_land_cover_table)):
        lc_rgb = df_land_cover_table.loc[i, 'color']

        # lc_rgb = np.array([df_land_cover_table.loc[i, 'R'], df_land_cover_table.loc[i, 'G'], df_land_cover_table.loc[i, 'B'], 255])

        color_table.SetColorEntry(i + 1, tuple(lc_rgb))

    # Assign the color table to the band
    band.SetRasterColorTable(color_table)

    # Save the changes and close the dataset
    dataset = None

    return None


# def main():
if __name__ == '__main__':

    path_vermont = join(rootpath, 'data', 'high_resolution_land_cover', 'Vermont')

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)
    proj_ard = gpd_ard.crs

    year = 2016

    filename_proj_ard_land_cover = join(path_vermont, 'ard_proj_land_cover', f'vermont_{year}_qgis.tif')

    filename_compress = join(path_vermont, 'ard_proj_land_cover', f'vermont_{year}.tif')

    vermont_save_with_compression(filename_proj_ard_land_cover, filename_compress, compression='LZW')   # compress the land cover

    ##

    # define the land cover table for Vermont land cover
    df_land_cover_table = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                        'land_cover': ['Tree Canopy', 'Grass/Shrubs', 'Bare Soil', 'Water',
                                                       'Buildings', 'Roads', 'Other Impervious', 'Railroads', 'Compacted Bare Soil'],
                                        'color': [(22, 230, 174, 255),  # 1 Tree Canopy
                                                  (202, 71, 148, 255),  # 2 Grass/Shrubs
                                                  (188, 68, 228, 255),  # 3 Bare Soil
                                                  (36, 20, 210, 255),  # 4 Water
                                                  (215, 181, 86, 255),  # 5 Buildings
                                                  (203, 53, 40, 255),  # 6 Roads
                                                  (129, 213, 135, 255),  # 7 Other Impervious
                                                  (174, 224, 100, 255),  # 8 Railroads
                                                  (131, 189, 227, 255),  # 9 Compacted Bare Soil
                                                  ]
                                        })

    vermont_add_pyramids_color_in_lc_tif(filename_compress, df_land_cover_table)



