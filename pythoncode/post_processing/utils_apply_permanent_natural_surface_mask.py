"""
    utility functions to apply the permanent natural surface mask and the US land boundary mask
"""

import numpy as np
from os.path import join, exists
import os
import sys
from osgeo import gdal_array

pwd = os.getcwd()
rootpath = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.util_function.tile_name_convert import convert_6_tile_names_to_8_tile_names


def apply_permanent_natural_surface_us_land_boundary_mask(img_stack_ts_is_pct_post_processing,
                                                          tile_name,
                                                          label_permanent_natural_surface=0,):
    """
    apply the permanent natural surface and the US land boundary mask to the post-processed ISP images

    :param img_stack_ts_is_pct_post_processing:
    :param tile_name: 8-tile name
    :param label_permanent_natural_surface: label for permanent natural surface, default is 0, for IS change types is 1 (stable natural surface)

    :return:
    """

    if len(tile_name) == 6:
        tile_name = convert_6_tile_names_to_8_tile_names(tile_name)

    # get the permanent natural surface mask
    filename_permanent_natural_surface_mask = join(rootpath, 'data', 'permanent_natural_surface_mask', 'individual_tile', tile_name,
                                                   f'permanent_natural_surface_count_v2_{tile_name}.tif')

    assert exists(filename_permanent_natural_surface_mask), f'{filename_permanent_natural_surface_mask} does not exist'
    img_permanent_natural_surface_count = gdal_array.LoadFile(filename_permanent_natural_surface_mask)
    mask_permanent_natural_surface = (img_permanent_natural_surface_count == 38)  # True for permanent natural surface

    # get the US land boundary mask
    filename_us_land_boundary = join(rootpath, 'data', 'shapefile', 'CONUS_boundary', f'tl_2023_us_state', 'individual_tile_state',
                                     f'tl_2023_us_state_{tile_name}.tif')
    assert exists(filename_us_land_boundary), f'{filename_us_land_boundary} does not exist'
    img_us_land_boundary = gdal_array.LoadFile(filename_us_land_boundary)

    mask_us_land = ~(img_us_land_boundary == 255)  # true for land, false for ocean

    # apply the permanent natural surface and the US land boundary mask
    for j_tmp in range(0, np.shape(img_stack_ts_is_pct_post_processing)[0]):
        img_stack_ts_is_pct_post_processing[j_tmp][~mask_us_land] = 255
        img_stack_ts_is_pct_post_processing[j_tmp][mask_permanent_natural_surface] = label_permanent_natural_surface

    return img_stack_ts_is_pct_post_processing







