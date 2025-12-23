"""
    predict the ISP with the trained UNet model
    
    For one core, predicting the 1985-2022 (38 years) ISP for one tile, it takes about 3-5 hourse, the avarage time is about 4 hours.
"""

from osgeo import gdal, gdalconst, gdal_array
import os
from os.path import join
import numpy as np
import time
import glob
import sys
import click
import pandas as pd
import logging
import yaml

import torch
from torch import from_numpy

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

from deep_learning_isp.unet_model import UNet
from deep_learning_isp.utils_deep_learning import (read_cold_variable, get_proj_info,
                                                   read_global_normalization_boundary, add_pyramids_color_in_nlcd_isp_tif)
from deep_learning_isp.prepare_training_sample_cb import predictor_normalize

from Basic_tools.Figure_plot import FP
from deep_learning_isp.unet_prediction import (pipe_line_unet_prediction)
from deep_learning_isp.mask_unet_regressor_isp_with_binary_mask import mask_unet_regressor_isp_with_binary_mask


@click.command()
@click.option('--rank', type=int, default='$SLURM_ARRAY_TASK_ID', help='the tile index, maximum is 27')
def main(rank):

# if __name__ == "__main__":
    # np.set_printoptions(precision=4, suppress=True)

    list_year = np.arange(1985, 2022, 1)
    # list_test_tile = ['h027v008', 'h028v008', 'h027v009']
    
    # list_tile_name = ['h003v002', 'h003v012', 'h007v013', 'h016v014', 'h021v007',
    #                   'h020v016', 'h025v017', 'h027v008', 'h027v009', 
    #                   'h028v008', 'h028v009',
    #                   'h029v004',
    #                   'h030v006']
    
    # list_tile_name = ['h001v008', 'h004v001', 'h008v008', 'h017v017',
    #                   'h018v005', 'h020v010', 'h021v017', 'h027v007',
    #                   'h028v010', 'h029v007',]
    
    list_tile_name = ['h003v002', 'h003v012', 'h007v013', 'h016v014', 'h021v007',
                      'h020v016', 'h025v017', 'h027v008', 'h027v009', 'h028v008', 
                      'h028v009', 'h029v004', 'h030v006', 'h001v008', 'h004v001', 
                      'h008v008', 'h017v017', 'h018v005', 'h020v010', 'h021v017', 
                      'h027v007', 'h028v010', 'h029v007',]
    
    # binary_classification_v1, binary_classification_conus_v1
    # for training_version in 'classifier_v1', 'regressor_v1', 'regressor_v2_conus', 'binary_classification_conus_v1', binary_classification_v4_conus_topography, regressor_v4_conus_topography
    # regressor_v5_conus_topography
    
    central_reflectance_flag = 'change'
    norm_boundary_folder='maximum_minimum_ref_conus'
    
    epoch_reg = 98
    epoch_cls = 100
    
    training_version_isp_regression = 'regressor_v6_conus_topography'  # version of the training model, e.g., 'classifier_v1', 'regressor_v1', 'regressor_v2_conus'
    isp_regression_folder = f'epoch{epoch_reg:03d}_predict_isp_change'
    filename_prefix_isp_regression = 'unet_regressor_round'

    training_version_is_mask = 'binary_classification_v6_conus_topography'   # 'binary_classification_conus_v1', 'binary_classification_v1'
    is_mask_folder = f'epoch{epoch_cls:03d}_predict_is_mask'
    filename_prefix_is_mask = 'unet_classifier'
    
    filename_prefix_mask_isp_output = 'unet_regressor_round_masked'
    
    if rank > len(list_tile_name):
        print('{}: this is the last running rank'.format(rank))
    else:
        tile_name = list_tile_name[rank - 1]
        
        for i_year in range(0, len(list_year)):
            year = list_year[i_year]
        
            list_year_prediction = np.arange(year, year + 1)    # list of years for prediction
        
            pipe_line_unet_prediction(tile_name=tile_name, 
                                    list_year=list_year_prediction, 
                                    task_type_flag='classification', 
                                    training_version=training_version_is_mask, 
                                    epoch=epoch_cls, 
                                    central_reflectance_flag=central_reflectance_flag, 
                                    output_folder_name=is_mask_folder, 
                                    rootpath_project_folder=None,
                                    norm_boundary_folder=norm_boundary_folder)
                
            pipe_line_unet_prediction(tile_name=tile_name, 
                                    list_year=list_year_prediction, 
                                    task_type_flag='regression', 
                                    training_version=training_version_isp_regression,
                                    epoch=epoch_reg, 
                                    central_reflectance_flag=central_reflectance_flag, 
                                    output_folder_name=isp_regression_folder, 
                                    rootpath_project_folder=None,
                                    norm_boundary_folder=norm_boundary_folder)
                        
            # mask the original regression ISP output with the binary classification mask
            output_filename = mask_unet_regressor_isp_with_binary_mask(tile_name=tile_name, year=year,
                                                                        training_version_isp_regression=training_version_isp_regression, 
                                                                        isp_regression_folder=isp_regression_folder, 
                                                                        filename_prefix_isp_regression=filename_prefix_isp_regression,
                                                                        training_version_is_mask=training_version_is_mask, 
                                                                        is_mask_folder=is_mask_folder, 
                                                                        filename_prefix_is_mask=filename_prefix_is_mask,
                                                                        filename_prefix_mask_isp_output=filename_prefix_mask_isp_output,
                                                                        rootpath_project_folder=None,
                                                                        )

if __name__ == "__main__":
    main()

