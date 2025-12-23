"""
    add the color table in the clipped land cover raster for each Landsat ARD tiles
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
import rasterio
import logging

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def cb_add_pyramids_color_in_lc_tif(filename_compress, df_land_cover_table, list_overview=None):
    """
        add pyramids and color table in the vermont land cover tif file

        Adding the pyramids may not be necessary for Chesapeake Bay because the loading is already fluent
    """

    dataset = gdal.Open(filename_compress, gdal.GA_Update)

    # Generate overviews/pyramids
    # The list [2, 4, 8, 16, 32] defines the downsampling factors for the overviews

    # if list_overview is None:
    #     list_overview = [2, 4, 8, 16, 32, 64]
    #
    # dataset.BuildOverviews(overviewlist=list_overview)

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


def get_running_task_dataframe(df_2022_cb, path_2022_cb):
    """
    get the dataframe recording the running task

    :param df_2022_cb:
    :return:
    """

    df_running_task = pd.DataFrame(columns=['city_name', 'year', 'tile_name', 'file_name'])
    index = 0
    for i in range(0, len(df_2022_cb)):
        folder_name = df_2022_cb.loc[i, 'city_name']
        year = df_2022_cb.loc[i, 'year']

        list_file_name_high_res_lc = glob.glob(join(path_2022_cb, 'clip_ard_high_resolution', folder_name, f'{year}', '*.tif'))

        # print(i, folder_name, year, len(list_file_name_high_res_lc))

        for j in range(0, len(list_file_name_high_res_lc)):
            file_name_high_res_lc = list_file_name_high_res_lc[j]
            tile_name = os.path.split(file_name_high_res_lc)[-1][-10:-4]

            df_running_task.loc[index, 'city_name'] = folder_name
            df_running_task.loc[index, 'year'] = year
            df_running_task.loc[index, 'tile_name'] = tile_name
            df_running_task.loc[index, 'file_name'] = file_name_high_res_lc

            index += 1

    return df_running_task


# def main():
if __name__ == '__main__':

    path_2022_cb = join(rootpath, 'data', 'high_resolution_land_cover', '2022_Chesapeake_Bay')

    df_2022_cb = pd.read_excel(join(path_2022_cb, '2022_Chesapeake_Bay_table.xlsx'), sheet_name='Sheet1')

    df_running_task = get_running_task_dataframe(df_2022_cb, path_2022_cb)

    df_cb_land_cover_table = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                           'land_cover': ['Water', 'Emergent Wetlands',
                                                          'Tree Canopy', 'Scrub/Shrub',
                                                          'Low Vegetation', 'Barren',
                                                          'Impervious Structures', 'Other Impervious',
                                                          'Impervious Roads', 'Tree Canopy Over Structures',
                                                          'Tree Canopy Over Other Impervious', 'Tree Canopy Over Impervious Roads'],
                                           'color': [(0, 97, 255, 255),  # Water
                                                     (0, 168, 132, 255),  # Emergent Wetlands
                                                     (38, 115, 0, 255),  # Tree Canopy
                                                     (76, 230, 0, 255),  # Scrub/Shrub
                                                     (165, 245, 122, 255),  # Low Vegetation
                                                     (255, 170, 0, 255),  # Barren
                                                     (255, 0, 0, 255),  # Imperious Structures
                                                     (178, 178, 178, 255),  # Other Impervious
                                                     (0, 0, 0, 255),  # Impervious Roads
                                                     (115, 115, 0, 255),  # Tree Canopy Over Structures
                                                     (205, 205, 102, 255),  # Tree Canopy Over Other Impervious
                                                     (255, 255, 115, 255)  # Tree Canopy Over Impervious Roads
                                                     ]
                                           })

    ##
    # define the land cover table for Chesapeake Bay
    for i in range(0, len(df_running_task)):

        filename_land_cover = df_running_task.loc[i, 'file_name']
        print(filename_land_cover)

        cb_add_pyramids_color_in_lc_tif(filename_land_cover, df_cb_land_cover_table, list_overview=None)

    ##


