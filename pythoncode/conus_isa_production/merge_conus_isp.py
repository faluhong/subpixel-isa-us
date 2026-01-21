"""
    merge the CONUS ISP and IS change type images from individual tiles and years
    
    It takes ~3 minutes to finish one year ISP / IS change type image. The memory usage is ~60 GB.
    
    Estimated time to finish the whole process:
    1. ISP: 3 minutes * 38 years = 114 minutes
    2. IS change type: 3 minutes * 38 years = 114 minutes
    3. Total: 114 + 114 = 228 minutes = 3.8 hours
    
    Recommend to run with 96 GB memory and 10 CPU cores for parallel processing.
"""

import geopandas as gpd
import os
from os.path import join, exists
import sys
from osgeo import gdal, gdal_array, gdalconst
import numpy as np
import click
import re


pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


from pythoncode.model_training.utils_deep_learning import (add_pyramids_color_in_nlcd_isp_tif,
                                                           add_pyramids_color_in_is_change_type_tif)


def output_mosaic_isp_img(img_mosaiced_conus_isp, filename_mosaiced_conus_isp,
                          nrow=5000, ncol=5000, total_v=22, total_h=33, add_pyramids=True):
    """
    output the mosaiced ISP image

    :param img_mosaiced_conus_isp:
    :param nrow:
    :param ncol:
    :param total_v:
    :param total_h:
    :return:
    """

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)

    proj_ard = gpd_ard.crs.to_wkt()
    geo_transform = (gpd_ard.total_bounds[0], 30, 0, gpd_ard.total_bounds[3], 0, -30)

    if not os.path.exists(os.path.dirname(filename_mosaiced_conus_isp)):
        os.makedirs(os.path.dirname(filename_mosaiced_conus_isp), exist_ok=True)

    tif_output = gdal.GetDriverByName('GTiff').Create(filename_mosaiced_conus_isp, ncol * total_h, nrow * total_v, 1,
                                                      gdalconst.GDT_Byte, options=['COMPRESS=LZW'])
    tif_output.SetGeoTransform(geo_transform)
    tif_output.SetProjection(proj_ard)

    Band = tif_output.GetRasterBand(1)
    Band.WriteArray(img_mosaiced_conus_isp)

    tif_output = None
    del tif_output

    if add_pyramids:
        add_pyramids_color_in_nlcd_isp_tif(filename_mosaiced_conus_isp, list_overview=None)



def conus_mosaic_isp(filename_prefix,
                     folder_output,
                     list_predicted_tiles, 
                     year, 
                     nrow=5000, ncol=5000, total_v=22, total_h=33,
                     rootpath_project_folder=None,
                     ):
    """
    get the available isp pixel count in the CONUS ARD grid based on the input dataframe

    :param df_isp:
    :param nrow:
    :param ncol:
    :param total_v:
    :param total_h:
    :return:
    """

    if rootpath_project_folder is None:
        rootpath_project_folder = rootpath

    img_merged_isp = np.zeros((total_v * nrow, total_h * ncol), dtype=np.uint8) + 255

    count_nan_files = 0
    for i_tile in range(0, len(list_predicted_tiles)):
        tile_name = list_predicted_tiles[i_tile]
        # print(year, tile_name)

        # using regrex to find the h and v index
        matches = re.findall(r'\d+', tile_name)
        integers = [int(match) for match in matches]

        h_index = integers[0]
        v_index = integers[1]

        filename_isp_tif = join(rootpath_project_folder, 'results', 'conus_isp', 
                                folder_output, f'{year}', tile_name, 
                                f'{filename_prefix}_{tile_name}_{year}_isp.tif')
        print(filename_isp_tif)
        
        # filename_finished_flag = join(rootpath_project_folder, 'results', 'conus_isp', f'{year}', tile_name, 
                                    #   f'{year}_{tile_name}_finished.txt')
        if not exists(filename_isp_tif):
            print(f'{filename_isp_tif} does not exist')
            count_nan_files += 1
            continue

        img_isp = gdal_array.LoadFile(filename_isp_tif)
        
        img_merged_isp[v_index * nrow:(v_index + 1) * nrow, h_index * ncol:(h_index + 1) * ncol] = img_isp

    return (img_merged_isp, count_nan_files)


def conus_mosaic_is_change_types(folder_output,
                                 list_predicted_tiles,
                                 year,
                                 nrow=5000, ncol=5000, total_v=22, total_h=33,
                                 rootpath_project_folder=None,
                                 ):

    """
    get the available isp pixel count in the CONUS ARD grid based on the input dataframe

    :param df_isp:
    :param nrow:
    :param ncol:
    :param total_v:
    :param total_h:
    :return:
    
    dict_is_change_type = {'1': 'stable natural',
                            '2': 'stable IS',
                            '3': 'IS expansion',
                            '4': 'IS intensification',
                            '5': 'IS decline',
                            '6': 'IS reversal',
                            '7': 'surface modification'}
    
    """

    if rootpath_project_folder is None:
        rootpath_project_folder = rootpath

    img_merged_is_change_type = np.zeros((total_v * nrow, total_h * ncol), dtype=np.uint8) + 255

    count_nan_files = 0
    for i_tile in range(0, len(list_predicted_tiles)):
        tile_name = list_predicted_tiles[i_tile]
        # print(year, tile_name)

        # using regrex to find the h and v index
        matches = re.findall(r'\d+', tile_name)
        integers = [int(match) for match in matches]

        h_index = integers[0]
        v_index = integers[1]

        filename_is_change_type_tif = join(rootpath_project_folder, 'results', 'conus_isp', 
                                folder_output, f'{year}-{year+1}', tile_name, 
                                f'{tile_name}_{folder_output}_{year}_{year + 1}_is_change_type.tif')
        print(filename_is_change_type_tif)
        
        # filename_finished_flag = join(rootpath_project_folder, 'results', 'conus_isp', f'{year}', tile_name, 
                                    #   f'{year}_{tile_name}_finished.txt')
        if not exists(filename_is_change_type_tif):
            print(f'{filename_is_change_type_tif} does not exist')
            count_nan_files += 1
            continue

        img_isp = gdal_array.LoadFile(filename_is_change_type_tif)
        
        img_merged_is_change_type[v_index * nrow:(v_index + 1) * nrow, h_index * ncol:(h_index + 1) * ncol] = img_isp
    
    return (img_merged_is_change_type, count_nan_files)



def output_mosaic_is_change_type_img(img_mosaiced_conus_isp, filename_mosaiced_conus_isp, 
                                     nrow=5000, ncol=5000, total_v=22, total_h=33, 
                                     flag_add_colors=True, gdal_type=gdalconst.GDT_Byte):
    """
    output the merged CONUS IS change type image
    :return:
    """

    filename_conus_ard_grid = join(rootpath, 'data', 'shapefile', 'CONUS_ARD', 'conus_ard_grid.shp')
    gpd_ard = gpd.read_file(filename_conus_ard_grid)

    proj_ard = gpd_ard.crs.to_wkt()
    geo_transform = (gpd_ard.total_bounds[0], 30, 0, gpd_ard.total_bounds[3], 0, -30)

    if not os.path.exists(os.path.dirname(filename_mosaiced_conus_isp)):
        os.makedirs(os.path.dirname(filename_mosaiced_conus_isp), exist_ok=True)

    tif_output = gdal.GetDriverByName('GTiff').Create(filename_mosaiced_conus_isp, ncol * total_h, nrow * total_v, 1,
                                                      gdal_type, options=['COMPRESS=LZW'])
    tif_output.SetGeoTransform(geo_transform)
    tif_output.SetProjection(proj_ard)

    Band = tif_output.GetRasterBand(1)
    Band.WriteArray(img_mosaiced_conus_isp)

    tif_output = None
    del tif_output
    
    if flag_add_colors:
    
        # add pyramids and color table to the output image
        colors = np.array([np.array([108, 169, 102, 255]) / 255,  # stable natural
                        np.array([179, 175, 164, 255]) / 255,  # stable IS
                        np.array([255, 0, 0, 255]) / 255,  # IS expansion
                        np.array([126, 30, 156, 255]) / 255,  # IS intensification
                        np.array([250, 192, 144, 255]) / 255,  # IS decline
                        np.array([29, 101, 51, 255]) / 255,  # IS reversal
                        np.array([130, 201, 251, 255]) / 255,  # Surface modification
                        ])

        add_pyramids_color_in_is_change_type_tif(filename_mosaiced_conus_isp, list_overview=None, colors=colors)
   

@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
# if __name__ =='__main__':
    # rank = 1
    # n_cores = 2000

    rootpath_project_folder = r'/shared/zhulab/Falu/CSM_project/'   # rootpath # r'/shared/zhulab/Falu/CSM_project/'
    # rootpath_project_folder = rootpath   # rootpath # r'/shared/zhulab/Falu/CSM_project/'
    
    isp_folder = 'individual_year_tile_post_processing_is_expansion_ndvi015_sm'    # ISP folder to store the individual year tiles
    filename_prefix = 'unet_regressor_round_masked_post_processing'     # filename prefix of the individual year tiles
    
    output_folder_merged_conus_isp = 'merge_conus_isp_post_processing_is_expansion_ndvi015_sm' # folder to store the merged conus isp
    filename_output = 'conus_isp_post_processing_is_expansion_ndvi015_sm'   # filename prefix of the merged CONUS ISP 
    
    # isp_folder = 'individual_year_tile_post_processing_mean_ndvi010'    # ISP folder to store the individual year tiles
    # filename_prefix = 'unet_regressor_round_masked_post_processing'     # filename prefix of the individual year tiles
    
    # output_folder_merged_conus_isp = 'merge_conus_isp_post_processing_mean_ndvi010' # folder to store the merged conus isp
    # filename_output = 'conus_isp_post_processing_mean_ndvi010'   # filename prefix of the merged CONUS ISP 

    list_year = np.arange(1985, 2023)
    # list_year = np.array([2015, 2020, 2022])
    # list_year = np.array([2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2021])
    # list_year = np.array([2001, 2002, 2003, 2004, 2006, 2007, 2008, 2009])
    # list_year = np.array([1986, 1987, 1988, 1989, 1991, 1992, 1993, 1994, 1996, 1997, 1998, 1999])

    each_core_block = int(np.ceil(len(list_year) / n_cores))
    for i in range(0, each_core_block):
        new_rank = rank - 1 + i * n_cores
        if new_rank > len(list_year) - 1:  # means that all folder has been processed
            print(f'{new_rank} this is the last running task')
        else:
            year = list_year[new_rank]
            print(f'{new_rank} {year} this is the running task')

            # CONUS ISP
            print(f'CONUS ISP {year}')

            list_predicted_tiles = os.listdir(join(rootpath_project_folder, 'results', 'conus_isp', isp_folder, f'{year}'))
            print(f'{len(list_predicted_tiles)} tiles are found in {year}')

            img_merged_isp, count_nan_files = conus_mosaic_isp(filename_prefix=filename_prefix,
                                                               folder_output=isp_folder,
                                                               list_predicted_tiles=list_predicted_tiles,
                                                               year=year,
                                                               nrow=5000, ncol=5000, total_v=22, total_h=33,
                                                               rootpath_project_folder=rootpath_project_folder,
                                                               )
            print(f'count_nan_files: {count_nan_files}')

            output_filename_merged_isp = join(rootpath_project_folder, 'results', 'conus_isp', output_folder_merged_conus_isp,
                                            f'{filename_output}_{year}.tif')
            
            output_mosaic_isp_img(img_merged_isp, output_filename_merged_isp, 
                                  nrow=5000, ncol=5000, total_v=22, total_h=33, add_pyramids=True)
                
            # CONUS IS change type
            print(f'CONUS IS change type {year}-{year + 1}')
            
            if year == 2022:
                # skip the last year
                print(f'skip the last year {year}')
                continue
            
            list_predicted_tiles = os.listdir(join(rootpath_project_folder, 'results', 'conus_isp', isp_folder, f'{year}-{year + 1}'))
            print(f'{len(list_predicted_tiles)} tiles are found in {year}-{year + 1}')

            img_merged_is_change_type, count_nan_files = conus_mosaic_is_change_types(folder_output=isp_folder,
                                                                                      list_predicted_tiles=list_predicted_tiles,
                                                                                      year=year,
                                                                                      nrow=5000, ncol=5000, total_v=22, total_h=33,
                                                                                      rootpath_project_folder=rootpath_project_folder,
                                                                                      )
            print(f'count_nan_files: {count_nan_files}')

            output_filename_merged_is_change_type = join(rootpath_project_folder, 'results', 'conus_isp', output_folder_merged_conus_isp,
                                                         f'{filename_output}_{year}_{year+1}_is_change_type.tif')
            
            output_mosaic_is_change_type_img(img_merged_is_change_type, output_filename_merged_is_change_type, 
                                             nrow=5000, ncol=5000, total_v=22, total_h=33)
        
        
if __name__ =='__main__':     
    main()