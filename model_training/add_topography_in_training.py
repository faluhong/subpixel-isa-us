"""
    add the topography information in the training samples
"""

import numpy as np
import time
import sys
import os
from os.path import join, exists
import time
import random
from tqdm import tqdm
import glob
import click
from osgeo import gdal, gdal_array, gdalconst
import pandas as pd
import re

pwd = os.getcwd()
rootpath = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP, FP_ISP


def read_global_topography_boundary():
    """
        read the CONUS topography boundary for normalization
        (1) elevation
        (2) slope
        (3) aspect
    """

    filename_topography_boundary_stats = join(rootpath, 'results', 'deep_learning', 'maximum_minimum_ref', 'conus_topography_boundary.csv')
    df_topography_boundary = pd.read_csv(filename_topography_boundary_stats)

    # the upper and lower boundary of the topography information is defined based on the global boundary, not the percentile
    array_upper_elevation = df_topography_boundary['elevation_max'].values[-1]
    array_lower_elevation = df_topography_boundary['elevation_min'].values[-1]
    
    array_upper_slope = df_topography_boundary['slope_max'].values[-1]
    array_lower_slope = df_topography_boundary['slope_min'].values[-1]
    
    array_upper_aspect = df_topography_boundary['aspect_max'].values[-1]
    array_lower_aspect = df_topography_boundary['aspect_min'].values[-1]
    
    return array_upper_elevation, array_lower_elevation, array_upper_slope, array_lower_slope, array_upper_aspect, array_lower_aspect
    

# def main():
if __name__ == '__main__':

    training_sample_output_folder = 'training_sample_conus_v1'

    nrow, ncol = 5000, 5000
    chip_size = 256

    (array_upper_elevation, array_lower_elevation, 
     array_upper_slope, array_lower_slope, 
     array_upper_aspect, array_lower_aspect) = read_global_topography_boundary()
    
    path_topography = join(rootpath, 'data', 'topography')
    
    path_x_train = join(rootpath, 'results', 'deep_learning', training_sample_output_folder, 'x_training')
    path_x_train_topography = join(rootpath, 'results', 'deep_learning', training_sample_output_folder, 'x_training_topography')    # folder to save the training samples with topography information
    if not exists(path_x_train_topography):
        os.makedirs(path_x_train_topography, exist_ok=True)
    
    list_x_train = glob.glob(join(path_x_train, '*.npy'))
    list_x_train.sort()
    
    pattern_tile_namae = r'h\d{2,3}v\d{3}'
    pattern_int = r'\d{4}'
    
    for i_x_train in range(0, len(list_x_train)):
    # for i_x_train in range(1, 2):
        
        filename_x_train = os.path.split(list_x_train[i_x_train])[-1]   # get the filename of the training sample
        
        tile_name = re.findall(pattern_tile_namae, filename_x_train)[0]  # get the tile name from the filename
        
        # get the row and column index from the filename
        matches_int = re.findall(pattern_int, filename_x_train)        
        list_int_values = [int(match) if match else None for match in matches_int]

        row_id = list_int_values[-2]
        col_id = list_int_values[-1]
        
        print(i_x_train, filename_x_train, tile_name, row_id, col_id)
        
        # read the topography data
        filename_dem = join(path_topography, tile_name, f'{tile_name}_dem.tif')
        filename_slope = join(path_topography, tile_name, f'{tile_name}_slope.tif')
        filename_aspect = join(path_topography, tile_name, f'{tile_name}_aspect.tif')
        
        img_dem = gdal_array.LoadFile(filename_dem)[row_id: row_id + chip_size, col_id: col_id + chip_size]
        img_slope = gdal_array.LoadFile(filename_slope)[row_id: row_id + chip_size, col_id: col_id + chip_size]
        img_aspect = gdal_array.LoadFile(filename_aspect)[row_id: row_id + chip_size, col_id: col_id + chip_size]
        
        # add one dimension to the topography data
        img_dem = np.expand_dims(img_dem, axis=0)
        img_slope = np.expand_dims(img_slope, axis=0)
        img_aspect = np.expand_dims(img_aspect, axis=0)
        
        # cutoff to the same boundary
        img_dem[img_dem < array_lower_elevation] = array_lower_elevation
        img_dem[img_dem > array_upper_elevation] = array_upper_elevation
        
        img_slope[img_slope < array_lower_slope] = array_lower_slope
        img_slope[img_slope > array_upper_slope] = array_upper_slope
        
        img_aspect[img_aspect < array_lower_aspect] = array_lower_aspect
        img_aspect[img_aspect > array_upper_aspect] = array_upper_aspect
        
        # normalization the topography information
        img_dem = (img_dem - array_lower_elevation) / (array_upper_elevation - array_lower_elevation)
        img_slope = (img_slope - array_lower_slope) / (array_upper_slope - array_lower_slope)
        img_aspect = (img_aspect - array_lower_aspect) / (array_upper_aspect - array_lower_aspect)
        
        # read the training sample
        img_x_train = np.load(list_x_train[i_x_train])
        
        # combine the topography data with the training sample
        img_x_train_topography = np.concatenate([img_x_train, img_dem, img_slope, img_aspect], axis=0)
        
        # save the training sample with topography information
        filename_x_train_topography = join(path_x_train_topography, filename_x_train.replace('.npy', '_topography.npy'))
        np.save(filename_x_train_topography, img_x_train_topography)
        
        
        
        
        
        
        
        
        
        
    