"""
    resample the CONUS ISP data to a coarser grid for better visualization

    Test different resample method to see which one works best
    
    It requires 128 GB memory to run the resampling 
"""

import numpy as np
from os.path import join, exists
import os
import sys
from osgeo import gdal, gdal_array
import geopandas as gpd
from osgeo import gdal, gdal_array, gdalconst
import click

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = os.path.join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from evaluation.utils_plot_isp import plot_isp_single

from conus_isp_analysis.utils_load_merge_conus_isp import load_merge_conus_is
from Basic_tools.add_pyramids import (add_pyramids_in_tif)
from auxiliary_data_process.utils_get_us_land_boundary import (get_us_proj_information)


def output_cal_tif_results(filename_output, conus_resample_image, scale_factor, gdal_type=gdal.GDT_Float32):
    """
        output the calculated/resamples results to GeoTIFF files.

        :param filename_output: output filename
        :param conus_resample_image:
        :param scale_factor: resample scale factor
        :return:
    """

    (proj, geo_transform) = get_us_proj_information()
    geo_transform_update = (geo_transform[0], 30 * scale_factor, 0, geo_transform[3], 0, -30 * scale_factor)

    tif_output = gdal.GetDriverByName('GTiff').Create(filename_output,
                                                      np.shape(conus_resample_image)[1],
                                                      np.shape(conus_resample_image)[0], 
                                                      1,
                                                      gdal_type,
                                                      options=['COMPRESS=LZW'])
    tif_output.SetGeoTransform(geo_transform_update)
    tif_output.SetProjection(proj)

    Band = tif_output.GetRasterBand(1)
    Band.WriteArray(conus_resample_image)

    tif_output = None
    del tif_output

    add_pyramids_in_tif(filename_output, resampling_methods='AVERAGE')


@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
# if __name__ == '__main__':

    output_folder_merged_conus_isp = 'merge_conus_isp_post_processing_binary_is_ndvi015_sm'  # folder to store the merged conus isp
    output_filename_prefix = 'conus_isp_post_processing_binary_is_ndvi015_sm'

    scale_factor = 33  # resample factor
    print(f'resample scale factor: {scale_factor}, i.e., {30 * scale_factor} m resolution')
    
    array_year = np.arange(1988, 2021)
    
    # save the resampled image
    output_folder = join(rootpath, 'results', 'conus_isp', 'resampled_conus_isp', f'resample_{30 * scale_factor}m')
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    each_core_block = int(np.ceil(len(array_year) / n_cores))
    for i in range(0, each_core_block):
        new_rank = rank - 1 + i * n_cores
        # means that all folder has been processed
        if new_rank > len(array_year) - 1:
            print(f'{new_rank} this is the last running task')
        else:
    
            year = array_year[new_rank]
            print(f'Processing year: {year}')

            img_conus_isp = load_merge_conus_is(output_folder_merged_conus_isp, output_filename_prefix, year, data_type='isp',)

            img_conus_isp = img_conus_isp.astype(np.float32)
            img_conus_isp[img_conus_isp == 255] = np.nan  # set no data values to nan for resampling

            # resample to a coarser grid
            # assert img_conus_isp.shape[0] % scale_factor == 0, "Input image height must be divisible by scale factor"
            # assert img_conus_isp.shape[1] % scale_factor == 0, "Input image width must be divisible by scale factor"
            
            nrows_coarse = img_conus_isp.shape[0] // scale_factor
            ncols_coarse = img_conus_isp.shape[1] // scale_factor

            img_conus_isp_coarse = img_conus_isp[0:nrows_coarse * scale_factor, 0:ncols_coarse * scale_factor].reshape(
                nrows_coarse, scale_factor, ncols_coarse, scale_factor).mean(axis=(1, 3))

            # output the resampled tif file
            filename_output = join(output_folder, f'{output_filename_prefix}_{year}_{30 * scale_factor}m.tif')

            output_cal_tif_results(filename_output=filename_output,
                                   conus_resample_image=img_conus_isp_coarse,
                                   scale_factor=scale_factor)


if __name__ == '__main__':
    main()  





