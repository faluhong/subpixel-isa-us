"""
    calculate the pixel-level reduction and recovery rates for ISPs during financial crises

    The ISP changes between 2005 and 2008 should be larger than 0.01% to avoid division by zero issues.
"""

import numpy as np
import os
from os.path import join, exists
import sys
import pandas as pd
from osgeo import gdal, gdal_array, gdalconst
import geopandas as gpd
import matplotlib.pyplot as plt

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

# from Basic_tools.Figure_plot import FP
# from Basic_tools.utils_hist_bar_plot import hist_plot_stats
from analysis.resample_conus_isp import (output_cal_tif_results)


def read_resampled_isp_data(year, scale_factor):
    """
    Read the resampled ISP data from a GeoTIFF file.
    """

    output_folder_resampled = join(rootpath, 'results', 'conus_isp', 'resampled_conus_isp', f'resample_{30*scale_factor}m')

    output_filename_prefix = 'conus_isp_post_processing_binary_is_ndvi015_sm'

    output_raster = join(output_folder_resampled, f'{output_filename_prefix}_{year}_{30 * scale_factor}m.tif')

    img_data = gdal_array.LoadFile(output_raster)

    return img_data


# def main():
if __name__ =='__main__':

    # 30, 34, 100, 200, 300, 3000
    scale_factor = 100   # scale factor used during resampling

    # read the resampled ISP data, the unit is %
    conus_isp_resampled_2005 = read_resampled_isp_data(2005, scale_factor)
    conus_isp_resampled_2006 = read_resampled_isp_data(2006, scale_factor)
    conus_isp_resampled_2007 = read_resampled_isp_data(2007, scale_factor)
    conus_isp_resampled_2008 = read_resampled_isp_data(2008, scale_factor)
    conus_isp_resampled_2009 = read_resampled_isp_data(2009, scale_factor)
    conus_isp_resampled_2010 = read_resampled_isp_data(2010, scale_factor)
    conus_isp_resampled_2011 = read_resampled_isp_data(2011, scale_factor)

    conus_isp_resampled_2017 = read_resampled_isp_data(2017, scale_factor)
    conus_isp_resampled_2018 = read_resampled_isp_data(2018, scale_factor)
    conus_isp_resampled_2019 = read_resampled_isp_data(2019, scale_factor)
    conus_isp_resampled_2020 = read_resampled_isp_data(2020, scale_factor)

    conus_isp_change_2005_2008 = (conus_isp_resampled_2008 - conus_isp_resampled_2005) / 3.0
    conus_isp_change_2008_2011 = (conus_isp_resampled_2011 - conus_isp_resampled_2008) / 3.0
    conus_isp_change_2017_2020 = (conus_isp_resampled_2020 - conus_isp_resampled_2017) / 3.0

    ##
    # divide the absolution ISP change during 2005 to 2008
    conus_reduction_pct = (conus_isp_change_2008_2011 - conus_isp_change_2005_2008) / np.abs(conus_isp_change_2005_2008) * 100
    conus_recovery_pct = conus_isp_change_2017_2020 / np.abs(conus_isp_change_2005_2008) * 100

    conus_resilience_pct = conus_recovery_pct - conus_reduction_pct

    # set the inf values (divided by zero) to nan
    conus_reduction_pct[np.isinf(conus_reduction_pct)] = np.nan
    conus_recovery_pct[np.isinf(conus_recovery_pct)] = np.nan
    conus_resilience_pct[np.isinf(conus_resilience_pct)] = np.nan

    ##
    # output_folder = join(rootpath, 'results', 'conus_is_product_visualization', f'isp_pct_change',
    #                      f'resample_{30*scale_factor}m')
    # if not exists(output_folder):
    #     os.makedirs(output_folder, exist_ok=True)
    #
    # filename_isp_change_2005_2008 = join(output_folder, f'conus_isp_change_2005_2008_{30 * scale_factor}m.tif')
    # output_cal_tif_results(filename_output=filename_isp_change_2005_2008,
    #                        conus_resample_image=conus_isp_change_2005_2008,
    #                        scale_factor=scale_factor, )
    #
    # filename_isp_change_2008_2011 = join(output_folder, f'conus_isp_change_2008_2011_{30 * scale_factor}m.tif')
    # output_cal_tif_results(filename_output=filename_isp_change_2008_2011,
    #                        conus_resample_image=conus_isp_change_2008_2011,
    #                        scale_factor=scale_factor, )
    #
    # filename_isp_change_2017_2020 = join(output_folder, f'conus_isp_change_2017_2020_{30 * scale_factor}m.tif')
    # output_cal_tif_results(filename_output=filename_isp_change_2017_2020,
    #                        conus_resample_image=conus_isp_change_2017_2020,
    #                        scale_factor=scale_factor, )
    #
    # filename_reduction = join(output_folder, f'conus_reduction_{30 * scale_factor}m.tif')
    # output_cal_tif_results(filename_output=filename_reduction,
    #                        conus_resample_image=conus_reduction_pct,
    #                        scale_factor=scale_factor,)
    #
    # filename_recovery = join(output_folder, f'conus_recovery_{30 * scale_factor}m.tif')
    # output_cal_tif_results(filename_output=filename_recovery,
    #                        conus_resample_image=conus_recovery_pct,
    #                        scale_factor=scale_factor,)
    #
    # filename_resilience = join(output_folder, f'conus_resilience_{30 * scale_factor}m.tif')
    # output_cal_tif_results(filename_output=filename_resilience,
    #                        conus_resample_image=conus_resilience_pct,
    #                        scale_factor=scale_factor,)





