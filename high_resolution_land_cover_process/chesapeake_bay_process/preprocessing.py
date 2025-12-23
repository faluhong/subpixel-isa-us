import os
import sys
import fiona
import glob
import numpy as np
from osgeo import gdal, ogr, gdal_array, gdalconst
from os.path import join
import time
import click

pwd = os.getcwd()
rootpath_project = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)
print('pwd:', pwd)
print('rootpath project:', rootpath_project)
print('path code:', path_pythoncode)

from CSM_running.CSM_running_function import get_projection_info

NRows, NCols = 5000, 5000
Pixel_Size = 30

def clip_cb_landcover(output_rootpath, tilename, cb_filename):
    """
    clip the Chesapeake land cover data
    Args:
        output_rootpath: the output rootpath
        tilename:
        cb_filename: the filename of 1-meter Chesapeake Bay land cover
    Returns:
        img_lc_2013
    """

    filename_shp = join(rootpath_project, 'data', 'CB_ARDtile_shp', '{}.shp'.format(tilename))
    print(filename_shp)

    output_path = join(output_rootpath, tilename)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_filename = join(output_path, 'CB_' + tilename + '_13class.tif')

    res = 1

    shp = fiona.open(filename_shp)
    bds = shp.bounds

    ll = (bds[0], bds[1])
    ur = (bds[2], bds[3])
    coords = list(ll + ur)
    xmin_shp, xmax_shp, ymin_shp, ymax_shp = coords[0], coords[2], coords[1], coords[3]
    width, height = (coords[2] - coords[0]) // res, (coords[3] - coords[1]) // res

    dst_proj = 'PROJCS["Albers_Conic_Equal_Area",GEOGCS["WGS 84",DATUM["WGS_1984",' \
               'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
               'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],' \
               'PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],' \
               'PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],' \
               'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

    params = gdal.WarpOptions(format='GTiff', dstSRS=dst_proj,
                              outputBounds=[xmin_shp, ymin_shp, xmax_shp, ymax_shp], xRes=res, yRes=res,
                              resampleAlg=gdal.GRIORA_NearestNeighbour)
    dst = gdal.Warp(destNameOrDestDS=output_filename, srcDSOrSrcDSTab=cb_filename, options=params)
    img_lc_2013 = dst.ReadAsArray()

    dst = None

    return img_lc_2013


def isp_calculation(lc_cb_2013):
    """
    calculate the ISP
    Args:
        lc_cb_2013: array of Chesapeake Bay land cover data
    Returns:
        img_cb_isp_without_tree: isp image excluding the 'tree over impervious surface' category
        img_cb_isp_with_tree: isp image including the 'tree over impervious surface' category
    """

    nrow_cb_1m, ncol_cb_1m = Pixel_Size * NRows, Pixel_Size * NCols

    img_cb_isp_without_tree = np.zeros((NRows, NCols), dtype=np.float32)
    img_cb_isp_with_tree = np.zeros((NRows, NCols), dtype=np.float32)

    for row_id in range(0, nrow_cb_1m, Pixel_Size):
        for col_id in range(0, ncol_cb_1m, Pixel_Size):

            if (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 0).all():
                img_cb_isp_without_tree[row_id // Pixel_Size, col_id // Pixel_Size] = np.nan
                img_cb_isp_with_tree[row_id // Pixel_Size, col_id // Pixel_Size] = np.nan
                # print(row_id, col_id, row_id // Pixel_Size, col_id // Pixel_Size, 'all nan')
            else:
                isp_count_without_tree = np.count_nonzero(
                    (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 7)
                    | (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 8)
                    | (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 9)
                )
                isp_pecentage_without_tree = isp_count_without_tree / Pixel_Size / Pixel_Size
                img_cb_isp_without_tree[row_id // Pixel_Size, col_id // Pixel_Size] = isp_pecentage_without_tree

                isp_count_with_tree = np.count_nonzero(
                    (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 7)
                    | (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 8)
                    | (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 9)
                    | (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 10)
                    | (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 11)
                    | (lc_cb_2013[row_id:row_id + Pixel_Size, col_id:col_id + Pixel_Size] == 12)
                )
                ISP_pecentage_with_tree = isp_count_with_tree / Pixel_Size / Pixel_Size
                img_cb_isp_with_tree[row_id // Pixel_Size, col_id // Pixel_Size] = ISP_pecentage_with_tree

                # print(row_id, col_id, row_id // Pixel_Size, col_id // Pixel_Size,
                #       isp_count_without_tree, isp_pecentage_without_tree,
                #       isp_count_with_tree, ISP_pecentage_with_tree)

    img_cb_isp_without_tree = img_cb_isp_without_tree * 100
    img_cb_isp_with_tree = img_cb_isp_with_tree * 100

    return img_cb_isp_without_tree, img_cb_isp_with_tree


def output_isp(img_cb_isp_without_tree, img_cb_isp_with_tree, tilename, src_geotrans, src_proj):
    """
    output the ISP
    Args:
        img_cb_isp_without_tree:
        img_cb_isp_with_tree:
        tilename:
        src_geotrans:
        src_proj:
    Returns:
    """

    output_filename = join(rootpath_project, 'data', 'Chesapeake_Bay', 'LC_2013', tilename,
                           'isp_{}_13class_without_tree.tif'.format(tilename))

    tif_temp = gdal.GetDriverByName('GTiff').Create(output_filename, NCols, NRows, 1, gdalconst.GDT_Float32)
    tif_temp.SetGeoTransform(src_geotrans)
    tif_temp.SetProjection(src_proj)

    Band = tif_temp.GetRasterBand(1)
    Band.WriteArray(img_cb_isp_without_tree)

    del tif_temp


    output_filename = join(rootpath_project, 'data', 'Chesapeake_Bay', 'LC_2013', tilename,
                           'isp_{}_13class_with_tree.tif'.format(tilename))

    tif_temp = gdal.GetDriverByName('GTiff').Create(output_filename, NCols, NRows, 1, gdalconst.GDT_Float32)
    tif_temp.SetGeoTransform(src_geotrans)
    tif_temp.SetProjection(src_proj)

    Band = tif_temp.GetRasterBand(1)
    Band.WriteArray(img_cb_isp_with_tree)

    del tif_temp


@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-16')
@click.option('--cb_filename', type=str, default='/shared/cn450/Falu/Data/Chesapeake_Bay/LandCover/Baywide_13Class_20132014/Baywide_13Class_20132014.tif', help='the filename of CB land cover')
def main(rank, cb_filename):

    list_tilename = ['h026v006', 'h026v007', 'h026v008', 'h026v009', 'h026v010',
                     'h027v006', 'h027v007', 'h027v008', 'h027v009', 'h027v010',
                     'h028v005', 'h028v006', 'h028v007', 'h028v008', 'h028v009', 'h028v010']

    starttime = time.perf_counter()

    if rank > len(list_tilename):
        print('{}: this is the last running rank'.format(rank))
    else:
        tilename = list_tilename[rank-1]
        print(tilename)

        # cb_filename = r'K:\Data\Chesapeake_Bay\LandCover\Baywide_13Class_20132014\Baywide_13Class_20132014.tif'
        output_rootpath = join(rootpath_project, 'data', 'Chesapeake_Bay', 'LC_2013')

        clip_outputname = join(output_rootpath, tilename, 'CB_' + tilename + '_13class.tif')
        if os.path.exists(clip_outputname):
            print('{} already exists'.format(clip_outputname))
            lc_cb_2013 = gdal_array.LoadFile(clip_outputname)
        else:
            print('clip the Chesapeake Bay land cover data in {}'.format(tilename))
            lc_cb_2013 = clip_cb_landcover(output_rootpath, tilename, cb_filename)

        img_cb_isp_without_tree, img_cb_isp_with_tree = isp_calculation(lc_cb_2013)

        src_geotrans, src_proj = get_projection_info(tilename)

        output_isp(img_cb_isp_without_tree, img_cb_isp_with_tree, tilename, src_geotrans, src_proj)

    end_time = time.perf_counter()
    print('Running time:', end_time - starttime)


if __name__ == '__main__':
    main()