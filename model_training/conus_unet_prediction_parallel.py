"""
    This script is used to predict the ISP output for the whole CONUS region using the trained UNet model.
    
    The total tile number is 427. 
    Using 50 cores, it takes 1.5 hours to finish the prediction for one year
    Using 100 cores, it takes 0.5-1.5 hours to finish the prediction for one year
    
    The total output ISP size is 50-55 GBs. 
"""

from osgeo import gdal, gdalconst, gdal_array
import os
from os.path import join
import numpy as np
import glob
import sys
import click
import pandas as pd
import logging

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from deep_learning_isp.unet_prediction import pipe_line_unet_prediction
from deep_learning_isp.mask_unet_regressor_isp_with_binary_mask import mask_unet_regressor_isp_with_binary_mask


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is a symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
# if __name__ == "__main__":

    central_reflectance_flag = 'change'
    norm_boundary_folder='maximum_minimum_ref_conus'
    epoch_reg = 98
    epoch_cls = 100
    output_rootpath = '/gpfs/sharedfs1/zhulab/Falu/CSM_project/'
    
    list_year = np.arange(2022, 2023, 1)
    list_tile_name = os.listdir(join(output_rootpath, 'data', 'predictor_variable'))
    
    training_version_isp_regression = 'regressor_v6_conus_topography'  # version of the training model, e.g., 'classifier_v1', 'regressor_v1', 'regressor_v2_conus'
    isp_regression_folder = 'epoch98_predict_isp_change'
    filename_prefix_isp_regression = 'unet_regressor_round'

    training_version_is_mask = 'binary_classification_v6_conus_topography'   # 'binary_classification_conus_v1', 'binary_classification_v1'
    is_mask_folder = 'epoch100_predict_is_mask'
    filename_prefix_is_mask = 'unet_classifier'
    
    filename_prefix_mask_isp_output = 'unet_regressor_round_masked'

    each_core_block = int(np.ceil(len(list_tile_name) / n_cores))
    for i in range(0, each_core_block):

        new_rank = rank - 1 + i * n_cores

        if new_rank > len(list_tile_name) - 1:  # means that all folder has been processed
            print(f'{new_rank} this is the last running task')
        else:
            tile_name = list_tile_name[new_rank - 1]
            
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
                                        rootpath_project_folder=output_rootpath,
                                        norm_boundary_folder=norm_boundary_folder)
                
                pipe_line_unet_prediction(tile_name=tile_name, 
                                        list_year=list_year_prediction, 
                                        task_type_flag='regression', 
                                        training_version=training_version_isp_regression,
                                        epoch=epoch_reg, 
                                        central_reflectance_flag=central_reflectance_flag, 
                                        output_folder_name=isp_regression_folder, 
                                        rootpath_project_folder=output_rootpath,
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
                                                                           rootpath_project_folder=output_rootpath,
                                                                           )
                print(output_filename)
        
if __name__ == "__main__":
    main()