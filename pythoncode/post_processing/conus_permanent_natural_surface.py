"""
    get the permanent natural surface mask for the conus region based on the annual NLCD land cover data

    Calculate the count of the permanent natural surface pixels for each year from 1985 to 2022 and output
"""

import os
from os.path import join
import sys
from osgeo import gdal, gdal_array
import numpy as np

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def get_nlcd_conus_land_cover(year, path_nlcd_conus_land_cover=None):
    """
    get the nlcd isp for CONUS region

    :param year:
    :param path_nlcd_conus_land_cover:
    :return:
    """
    nrows_conus, nolcs_conus = 110000, 165000

    if path_nlcd_conus_land_cover is None:
        path_nlcd_conus_land_cover = r'K:\Data\NLCD\annual_nlcd\Annual_NLCD_LndCov_1985-2023_CU_C1V0'

    img_nlcd_isp_expand = np.zeros((nrows_conus, nolcs_conus), dtype=np.uint8) + 250

    # the original nlcd isp file is 105000 x 160000, but the mask and CONUS ISP is 110000 x 165000
    filename_nlcd_isp = join(path_nlcd_conus_land_cover, f'Annual_NLCD_LndCov_{year}_CU_C1V0.tif')

    img_nlcd_isp_expand[0: 110000 - 5000, 5000: 165000] = gdal_array.LoadFile(filename_nlcd_isp)

    return img_nlcd_isp_expand



def add_pyramids_in_tif(filename_tif, list_overview=None,  resampling_methods='NEAREST'):
    """
        add pyramids and color table in the tif file
    """

    if list_overview is None:
        list_overview = [2, 4, 8, 16, 32, 64]

    dataset = gdal.Open(filename_tif, gdal.GA_Update)

    # Generate overviews/pyramids
    # The list [2, 4, 8, 16, 32] defines the downsampling factors for the overviews
    dataset.BuildOverviews(resampling=resampling_methods, overviewlist=list_overview)

    dataset = None

    return None


# def main():
if __name__ =='__main__':

    # path_nlcd_conus_land_cover = None
    path_nlcd_conus_land_cover = r'/scratch/zhz18039/fah20002/CSM_project/data/NLCD_annual/Annual_NLCD_LndCov_1985-2023_CU_C1V0/'

    list_year = np.arange(1985, 2023)

    img_permanent_natural_surface = np.zeros((110000, 165000), dtype=np.uint8)

    for i_year in range(0, len(list_year)):
        year = list_year[i_year]
        print(year)

        img_nlcd_lc_expand = get_nlcd_conus_land_cover(year, path_nlcd_conus_land_cover=path_nlcd_conus_land_cover)

        # v1: version 1 is not enough to exclude all the commission errors
        # those land cover types are considered as permanent natural surface in the post-processing
        # 11 Open Water
        # 12 Perennial Ice/Snow;
        # 31 Barren Land (Rock/Sand/Clay);
        # 52 Shrub/Scrub;
        # 90 Woody Wetlands
        # 95 Emergent Herbaceous Wetlands
        # for other land cover types, the binary IS classification is used.

        # mask_natural_surface = ((img_nlcd_lc_expand == 11) | (img_nlcd_lc_expand == 12) | (img_nlcd_lc_expand == 31) |
        # (img_nlcd_lc_expand == 52) | (img_nlcd_lc_expand == 90) | (img_nlcd_lc_expand == 95))

        # v2
        # those land cover types are considered as permanent natural surface in the post-processing
        # 11 Open Water;
        # 12 Perennial Ice/Snow;
        # 31 Barren Land (Rock/Sand/Clay);
        # 42 Evergreen Forest
        # 52 Shrub/Scrub;
        # 71 Grassland/Herbaceous
        # 90 Woody Wetlands;
        # 95 Emergent Herbaceous Wetlands
        # for other land cover types, the binary IS classification is used.

        mask_natural_surface = ((img_nlcd_lc_expand == 11) | (img_nlcd_lc_expand == 12) | (img_nlcd_lc_expand == 31) | 
                                (img_nlcd_lc_expand == 42) | (img_nlcd_lc_expand == 52) | (img_nlcd_lc_expand == 71) |
                                (img_nlcd_lc_expand == 90) | (img_nlcd_lc_expand == 95))
        
        img_permanent_natural_surface = img_permanent_natural_surface + mask_natural_surface

    ##
    from pythoncode.conus_isa_production.merge_conus_isp import output_mosaic_isp_img

    filename_merged_isp = join(rootpath, 'data', 'permanent_natural_surface_mask',
                               f'conus_permanent_natural_surface_count_v2.tif')
    output_mosaic_isp_img(img_permanent_natural_surface, filename_merged_isp,
                          nrow=5000, ncol=5000, total_v=22, total_h=33, add_pyramids=False)

    add_pyramids_in_tif(filename_merged_isp)






