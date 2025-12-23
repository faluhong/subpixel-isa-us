"""
    summarize the ISP calculated from the high resolution land cover map
"""

import time
import shapely
import fiona
from pathlib import Path
import geopandas as gpd
import os
from os.path import join, exists
import sys
from osgeo import gdal, gdal_array, gdalconst
import pandas as pd
import glob
from pyproj import Proj, CRS, Transformer
import numpy as np
import time
import logging
import click
import re

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP


def get_file_info_2022_chesapeake_bay(rootpath_isp, source):
    """
    get the file information for the 2022 Chesapeake Bay ISP data

    :param rootpath_isp:
    :param source:
    :return:
    """

    if not isinstance(source, str):
        TypeError('source should be a string')
    if not isinstance(rootpath_isp, str):
        TypeError('rootpath_isp should be a string')

    list_folder = os.listdir(join(rootpath_isp, source, 'with_tree'))

    index = 0
    df_high_resolution_isp = pd.DataFrame(columns=['source', 'city_folder', 'tile_id', 'year', 'pixel_count', 'file_name'])

    for i_folder in range(0, len(list_folder)):
        list_folder_year = os.listdir(join(rootpath_isp, source, 'with_tree', list_folder[i_folder]))

        for i_year in range(0, len(list_folder_year)):
            year = list_folder_year[i_year]

            list_tif_file = glob.glob(join(rootpath_isp, source, 'with_tree', list_folder[i_folder], year, '*.tif'))

            for i_tif in range(0, len(list_tif_file)):

                pattern_tile = r'h\d{2}v\d{2}'
                grid_ref = re.search(pattern_tile, list_tif_file[i_tif])  # Using regex to find the grid reference
                tile_id = grid_ref.group(0)

                img_isp = gdal_array.LoadFile(list_tif_file[i_tif])
                if np.isnan(img_isp).all():
                    print(f'{source} {tile_id} {year} all nan')
                    pass
                else:
                    print(f'{source} {tile_id} {year} {np.sum(~np.isnan(img_isp))}')

                    df_high_resolution_isp.loc[index, 'source'] = source
                    df_high_resolution_isp.loc[index, 'city_folder'] = list_folder[i_folder]
                    df_high_resolution_isp.loc[index, 'tile_id'] = tile_id
                    df_high_resolution_isp.loc[index, 'year'] = year
                    df_high_resolution_isp.loc[index, 'pixel_count'] = np.sum(~np.isnan(img_isp))
                    df_high_resolution_isp.loc[index, 'file_name'] = list_tif_file[i_tif]

                    index += 1

    return df_high_resolution_isp


def get_file_info_general(rootpath_isp, source):

    if not isinstance(source, str):
        TypeError('source should be a string')
    if not isinstance(rootpath_isp, str):
        TypeError('rootpath_isp should be a string')

    list_folder = os.listdir(join(rootpath_isp, source))

    index = 0
    df_high_resolution_isp = pd.DataFrame(columns=['source', 'city_folder', 'tile_id', 'year', 'pixel_count', 'file_name'])

    for i_folder in range(0, len(list_folder)):

        list_tif_file = glob.glob(join(rootpath_isp, source, list_folder[i_folder], '*.tif'))

        for i_tif in range(0, len(list_tif_file)):

            pattern_tile = r'h\d{2}v\d{2}'
            grid_ref = re.search(pattern_tile, list_tif_file[i_tif])  # Using regex to find the tile_id
            tile_id = grid_ref.group(0)

            pattern_year = r'\d{4}'
            grid_ref = re.search(pattern_year, list_tif_file[i_tif])  # Using regex to find the year
            year = grid_ref.group(0)

            img_isp = gdal_array.LoadFile(list_tif_file[i_tif])
            if np.isnan(img_isp).all():
                print(f'{source} {tile_id} {year} all nan')
                pass
            else:
                print(f'{source} {tile_id} {year} {np.sum(~np.isnan(img_isp))}')

                df_high_resolution_isp.loc[index, 'source'] = source

                df_high_resolution_isp.loc[index, 'city_folder'] = list_folder[i_folder]

                df_high_resolution_isp.loc[index, 'tile_id'] = tile_id
                df_high_resolution_isp.loc[index, 'year'] = year
                df_high_resolution_isp.loc[index, 'pixel_count'] = np.sum(~np.isnan(img_isp))
                df_high_resolution_isp.loc[index, 'file_name'] = list_tif_file[i_tif]

                index += 1

    return df_high_resolution_isp


def get_file_info_vermont(rootpath_isp, source):

    if not isinstance(source, str):
        TypeError('source should be a string')
    if not isinstance(rootpath_isp, str):
        TypeError('rootpath_isp should be a string')

    index = 0
    df_high_resolution_isp = pd.DataFrame(columns=['source', 'city_folder', 'tile_id', 'year', 'pixel_count', 'file_name'])

    list_tif_file = glob.glob(join(rootpath_isp, source, '*.tif'))

    for i_tif in range(0, len(list_tif_file)):

        pattern_tile = r'h\d{2}v\d{2}'
        grid_ref = re.search(pattern_tile, list_tif_file[i_tif])  # Using regex to find the tile_id
        tile_id = grid_ref.group(0)

        pattern_year = r'\d{4}'
        grid_ref = re.search(pattern_year, list_tif_file[i_tif])  # Using regex to find the year
        year = grid_ref.group(0)

        img_isp = gdal_array.LoadFile(list_tif_file[i_tif])
        if np.isnan(img_isp).all():
            print(f'{source} {tile_id} {year} all nan')
            pass
        else:
            print(f'{source} {tile_id} {year} {np.sum(~np.isnan(img_isp))}')

            df_high_resolution_isp.loc[index, 'source'] = source
            df_high_resolution_isp.loc[index, 'city_folder'] = 'vermont'
            df_high_resolution_isp.loc[index, 'tile_id'] = tile_id
            df_high_resolution_isp.loc[index, 'year'] = year
            df_high_resolution_isp.loc[index, 'pixel_count'] = np.sum(~np.isnan(img_isp))
            df_high_resolution_isp.loc[index, 'file_name'] = list_tif_file[i_tif]

            index += 1

    return df_high_resolution_isp


# def main():
if __name__ =='__main__':

    list_sources = ['2022_Chesapeake_Bay', 'CCAP', 'EnviroAtlas', 'urban_watch', 'vermont']

    rootpath_isp = join(rootpath, 'data', 'ISP_from_high_res_lc')

    ##
    for i_source in range(0, len(list_sources)):
    # for i_source in range(4, 5):

        source = list_sources[i_source]
        print(source)

        if source == '2022_Chesapeake_Bay':
            df_isp_chesapeake_bay = get_file_info_2022_chesapeake_bay(rootpath_isp, source)
        elif source == 'CCAP':
            df_isp_ccap = get_file_info_general(rootpath_isp, source)
        elif source == 'EnviroAtlas':
            df_isp_enviro_atlas = get_file_info_general(rootpath_isp, source)
        elif source == 'urban_watch':
            df_isp_urban_watch = get_file_info_general(rootpath_isp, source)
        elif source == 'vermont':
            df_isp_vermont = get_file_info_vermont(rootpath_isp, source)

    ##
    df_isp_all = pd.concat([df_isp_chesapeake_bay, df_isp_ccap, df_isp_enviro_atlas, df_isp_urban_watch, df_isp_vermont], ignore_index=True)

    ##
    filename_summary = join(rootpath_isp, 'summary_high_resolution_isp.xlsx')
    df_isp_all.to_excel(filename_summary, index=False)

    ##

    # df_isp_all.groupby(['source', 'year']).sum()










