"""
    utility functions to load merged CONUS ISP data

    (1) Annual CONUS ISP dataset
    (2) Generated the merged CONUS ISP dataset
"""

import numpy as np
from os.path import join, exists
import os
import sys
from osgeo import gdal, gdal_array

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = os.path.join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def load_merge_conus_is(output_folder_merged_conus_isp, output_filename_prefix, year, data_type='isp', ):
    """
        load the merged CONUS IS percentage or IS change type

        :param output_folder_merged_conus_isp: folder to store the merged conus isp
        :param output_filename_prefix: filename prefix of the merged CONUS ISP
        :param year: year
        :param data_type: 'isp' or 'is_change_type'
        :return:
    """

    # output_folder_merged_conus_isp = 'merge_conus_isp_post_processing_binary_is_ndvi015_sm'  # folder to store the merged conus isp
    # filename_output = 'conus_isp_post_processing_binary_is_ndvi015_sm'  # filename prefix of the merged CONUS ISP

    if data_type == 'isp':
        filename_conus_is_dataset = join(rootpath, 'results', 'conus_isp', output_folder_merged_conus_isp,
                                     f'{output_filename_prefix}_{year}.tif')
    elif data_type == 'is_change_type':
        filename_conus_is_dataset = join(rootpath, 'results', 'conus_isp', output_folder_merged_conus_isp,
                                     f'{output_filename_prefix}_{year}_{year+1}_is_change_type.tif')
    else:
        raise ValueError('data_type must be isp or is_change_type')

    assert exists(filename_conus_is_dataset), f'{filename_conus_is_dataset} does not exist'

    img_conus_is_dataset = gdal_array.LoadFile(filename_conus_is_dataset)

    return img_conus_is_dataset


def load_merge_conus_annual_nlcd(year, path_environment='local_pc'):
    """
        load the merged CONUS Annual NLCD dataset

        :param year:
        :param path_environment:
        :return:
    """

    if path_environment == 'local_pc':
        rootpath_nlcd = r'K:\Data\NLCD\annual_nlcd\Annual_NLCD_FctImp_1985-2023_CU_C1V0'
    elif path_environment == 'hpc':
        rootpath_nlcd = r'/shared/zhulab/Falu/CSM_project/data/NLCD_annual/Annual_NLCD_FctImp_1985-2023_CU_C1V0/'
    else:
        raise ValueError('path_environment must be local_pc or hpc')

    filename_annual_nlcd_isp = join(rootpath_nlcd, f'Annual_NLCD_FctImp_{year}_CU_C1V0.tif')
    assert exists(filename_annual_nlcd_isp), f'{filename_annual_nlcd_isp} does not exist'

    # the original nlcd isp file is 105000 x 160000, but the mask and CONUS ISP is 110000 x 165000, so we need to expand the image
    nrows_conus, nolcs_conus = 110000, 165000

    img_annual_nlcd_isp = np.zeros((nrows_conus, nolcs_conus), dtype=np.uint8) + 250
    img_annual_nlcd_isp[0: 110000 - 5000, 5000: 165000] = gdal_array.LoadFile(filename_annual_nlcd_isp)

    return img_annual_nlcd_isp


