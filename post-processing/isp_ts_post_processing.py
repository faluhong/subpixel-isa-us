"""
    This script is used to post-process the output of the ISP time series analysis.

    Major steps:
    (1) Within each temporal segment, using the majority vote of binary classification
    to determine the segment belongs to natural surface or impervious surface.
        If equal number of pixels are classified as natural and impervious, then set the segment as natural surface.

    (2) If the segment is impervious surface, there are several ways to represent the whole segment.
       a: start of the segment
       b: maximum ISP value of the segment
       c: mean of the segment
       d: min of the segment
       e: median of the segment

    (3) For the years that do not have segment covering July-1st, using the nearest segment to represent the whole year.
        i.e., comparing the July-1st

    Potential rules for comparison:
    If the overall NDVI difference between two segments is less than 0.1,
    then consider no change between these two segments and link them together as one segment.
    Ref: Deng & Zhu CSM paper: https://www.sciencedirect.com/science/article/pii/S0034425718304590
"""

import time
import numpy as np
from os.path import join, exists
import scipy.io as scio
from datetime import datetime
import click
import logging

from astropy.modeling.projections import long_name
from geopandas import gpd
import os
import sys
import pandas as pd
import glob
from osgeo import gdal, ogr, gdal_array, gdalconst
import matplotlib
import matplotlib.pyplot as plt
from pyproj import Proj, CRS, Transformer
import seaborn as sns
from collections import Counter


pwd = os.getcwd()
rootpath = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP
from Basic_tools.mat_to_dataframe import mat_to_dataframe
from Basic_tools.datetime_datenum_convert import datenum_to_datetime_matlabversion, datetime_to_datenum_matlabversion
from evaluation.utils_evaluation import convert_8_tile_names_to_6_tile_names, convert_6_tile_names_to_8_tile_names

from deep_learning_isp.utils_deep_learning import (get_proj_info, add_pyramids_color_in_nlcd_isp_tif)
from deep_learning_isp.unet_prediction import predict_isp_output
from conus_isp_production.utils_get_conus_tile_name import get_conus_tile_name


def define_logger(filename_logger):
    """
        define the logger for CONUS isp post-processing
    """

    logger_isp_cal = logging.getLogger('conus_isp_post_processing')
    logger_isp_cal.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(filename_logger)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger_isp_cal.addHandler(file_handler)

    logger_isp_cal.addHandler(file_handler)

    return logger_isp_cal


def load_conus_isp_stack(list_year, tile_name,
                         filename_prefix='unet_regressor_round_masked',
                         isp_folder='individual_year_tile',
                         rootpath_conus_isp=None):
    """
    load the CONUS ISP time series

    :param list_year:
    :param tile_name: the 8-tile name
    :param filename_prefix: the prefix of the ISP filename, default is 'unet_regressor_round_masked',
                            other options include: 'unet_regressor_round', 'unet_classifier', unet_regressor_round_masked

    :param rootpath_conus_isp: the rootpath of the CONUS ISP

    :return: img_stack_isp_ts: the ISP time series image stack
    """

    if rootpath_conus_isp is None:
        rootpath_conus_isp = rootpath

    nrow, ncol = 5000, 5000
    img_stack_isp_ts = np.zeros((len(list_year), nrow, ncol), dtype=np.uint8)

    for i_year in range(0, len(list_year)):
        year = list_year[i_year]

        if len(tile_name) == 6:
            tile_name_8 = convert_6_tile_names_to_8_tile_names(tile_name)
        else:
            tile_name_8 = tile_name

        filename_isp_tif = join(rootpath_conus_isp, 'results', 'conus_isp', isp_folder,
                                f'{year}', tile_name_8,
                                f'{filename_prefix}_{tile_name_8}_{year}_isp.tif')
        img_isp = gdal_array.LoadFile(filename_isp_tif)

        assert exists(filename_isp_tif), f'{filename_isp_tif} does not exist'

        img_stack_isp_ts[i_year, :, :] = img_isp

    return img_stack_isp_ts


def load_cold_reccg(tile_name, row_id, path_cold_reccg: str):
    """
    load the COLD reccg data for a specific tile and row

    :param tile_name:
    :param row_id:
    :param path_cold_reccg:
    :return:
    """

    path_record_change = join(path_cold_reccg, tile_name, 'TSFitLine')
    file_name_ydata_training = join(path_record_change, 'record_change_r0{:04d}.mat'.format(row_id + 1))

    data = scio.loadmat(file_name_ydata_training, verify_compressed_data_integrity=False)
    mat_rec_cg = data['rec_cg']
    df_matfile = mat_to_dataframe(mat_rec_cg)

    return df_matfile


def find_modes(data):
    """
    find the modes of the data. In case of tie, return all the modes

    :param data:
    :return:
    """

    count = Counter(data)
    max_frequency = max(count.values())
    modes = [k for k, v in count.items() if v == max_frequency]

    return modes


def generate_post_processing_mask(array_t_start, array_t_end, mask_match, array_year):
    """
    generate the post-processing mask for the ISP time series
    The mask indicates which year belongs to which segment
    In more detail, the mask contains two parts:
    (1) valid segment mask: the years that are covered by the segment
    (2) seamless segment mask: the year belongs to which segment.
                                If the year is not covered by the segment, then it is determined by the distance to the nearest segments

    It takes about 2s to process one line (5000 pixels)

    :param array_t_start: the start time of the segment
    :param array_t_end: the end time of the segment
    :param mask_match: the mask to locate the position of the current pixel in the COLD reccg data
    :param array_year: the years to be processed, such as: array_year = np.arange(1985, 2023, 1)

    :return:
    """

    array_valid_segment_mask = np.zeros((len(array_year),), dtype=np.uint8)
    array_seamless_segment_mask = np.zeros((len(array_year),), dtype=np.uint8)

    if np.count_nonzero(mask_match) == 0:
        # print(f'no COLD reccg data in row {row_id} col {col_id}')
        pass
    else:
        # print(f'tile {tile_name} row {row_id} col {col_id} segment count {np.count_nonzero(mask_match)}')

        t_start = array_t_start[mask_match]
        t_end = array_t_end[mask_match]

        for i_year in range(0, len(array_year)):
            target_year = array_year[i_year]  # the target year to determine which segment the year belongs to
            target_datenum = datetime_to_datenum_matlabversion(datetime(target_year, 7, 1))

            if target_datenum < t_start[0]:
                # the target year is before the first segment
                array_valid_segment_mask[i_year] = 0
                array_seamless_segment_mask[i_year] = 1
            elif target_datenum > t_end[-1]:
                # the target year is after the last segment
                array_valid_segment_mask[i_year] = 0
                array_seamless_segment_mask[i_year] = np.count_nonzero(mask_match)
            else:
                # the target year is within the whole segment range
                # (1) within the segment; (2) between the segments but no COLD curves (distance-based determination)

                if np.count_nonzero(mask_match) == 1:
                    # only one segment
                    if (target_datenum >= t_start[0]) and (target_datenum <= t_end[0]):
                        # within the segment
                        array_valid_segment_mask[i_year] = 1
                        array_seamless_segment_mask[i_year] = 1
                    else:
                        ValueError('error in the segment determination')
                else:
                    # multiple segments
                    for i_seg in range(0, np.count_nonzero(mask_match) - 1):
                        t_start_seg_current = t_start[i_seg]
                        t_end_seg_current = t_end[i_seg]

                        t_start_seg_next = t_start[i_seg + 1]
                        t_end_seg_next = t_end[i_seg + 1]

                        if (target_datenum >= t_start_seg_current) and (target_datenum <= t_end_seg_current):
                            # within the current segment
                            array_valid_segment_mask[i_year] = i_seg + 1
                            array_seamless_segment_mask[i_year] = i_seg + 1
                            break
                        elif (target_datenum > t_end_seg_current) and (target_datenum < t_start_seg_next):
                            # between the current and next segments
                            array_valid_segment_mask[i_year] = 0

                            # the value in this year is determined by the distance to the two segments
                            distance_previous_segment = np.abs(target_datenum - t_end_seg_current)
                            distance_next_segment = np.abs(target_datenum - t_start_seg_next)

                            if distance_previous_segment < distance_next_segment:
                                array_seamless_segment_mask[i_year] = i_seg + 1
                            else:
                                array_seamless_segment_mask[i_year] = i_seg + 2
                            break
                        elif (target_datenum >= t_start_seg_next) and (target_datenum <= t_end_seg_next):
                            # within the next segments, this is to handle the last segment
                            array_valid_segment_mask[i_year] = i_seg + 2
                            array_seamless_segment_mask[i_year] = i_seg + 2

    return (array_valid_segment_mask, array_seamless_segment_mask)


def post_processing_time_series(array_is_pct, array_is_mask, array_valid_segment_mask, array_seamless_segment_mask, post_processing_rule='mean'):
    """
    Post-processing the ISP time series data
    Rules for each segment
    (1) Binary classification: majority vote
    (2) Regression: different rules defined by post_processing_rule flag, default is mean ISP value

    :param array_is_pct: original ISP time series data
    :param array_is_mask: original IS mask
    :param array_valid_segment_mask:
    :param array_seamless_segment_mask:
    :return:
    """

    array_is_pct_post_processing = array_is_pct.copy()
    array_is_mask_post_processing = array_is_mask.copy()

    list_segment_id = np.unique(array_valid_segment_mask)

    for i_seg in range(0, len(list_segment_id)):
        seg_id = list_segment_id[i_seg]

        mask_seg_valid = array_valid_segment_mask == seg_id  # the valid segment mask, used for determining the segment
        mask_seg_seamless = array_seamless_segment_mask == seg_id  # the seamless segment mask, used for updating the values

        # rule for the classifier: majority vote
        is_mask_seg = array_is_mask[mask_seg_valid]
        is_mask_seg_modes = find_modes(is_mask_seg)  # the modes of the segment

        if len(is_mask_seg_modes) == 1:
            array_is_mask_post_processing[mask_seg_seamless] = is_mask_seg_modes[0]
        else:
            # equal number of natural and impervious
            # set the segment as natural surface
            array_is_mask_post_processing[mask_seg_seamless] = 0

        is_pct_seg = array_is_pct[mask_seg_valid]

        if post_processing_rule == 'max':
            # rule for the regressor: maximum ISP value
            is_pct_seg_post_processing = np.nanmax(is_pct_seg)
        elif post_processing_rule == 'start':
            # rule for the regressor: the start of the segment
            is_pct_seg_post_processing = is_pct_seg[0]
        elif post_processing_rule == 'mean':
            # rule for the regressor: the mean of the segment
            is_pct_seg_post_processing = np.round(np.nanmean(is_pct_seg))  # round to the nearest integer
        elif post_processing_rule == 'min':
            # rule for the regressor: the minimum of the segment
            is_pct_seg_post_processing = np.round(np.nanmin(is_pct_seg))  # round to the nearest integer
        elif post_processing_rule == 'median':
            # rule for the regressor: the median of the segment
            is_pct_seg_post_processing = np.round(np.nanmedian(is_pct_seg))  # round to the nearest integer
        else:
            is_pct_seg_post_processing = None
            raise ValueError('error in the post-processing flag')

        array_is_pct_post_processing[mask_seg_seamless] = is_pct_seg_post_processing

        # apply the IS mask to the IS pct
        array_is_pct_post_processing[array_is_mask_post_processing == 0] = 0

    return array_is_pct_post_processing


def pipeline_post_processing_isp_ts(img_stack_ts_is_pct, img_stack_ts_is_mask,
                                    array_year, path_cold_reccg,
                                    post_processing_rule,
                                    tile_name, rows_running_start, cols_running_start,
                                    rows_running_end, cols_running_end,
                                    logger):
    """
    pipeline for post-processing the ISP time series

    :param img_stack_ts_is_pct:
    :param img_stack_ts_is_mask:
    :param array_year:
    :param path_cold_reccg:
    :param tile_name:
    :param rows_running_start:
    :param cols_running_start:
    :param rows_running_end:
    :param cols_running_end:
    :param logger:
    :return:
    """


    img_stack_ts_is_pct_post_processing = img_stack_ts_is_pct.copy()

    logger.info(f'tile: {tile_name} rows: [{rows_running_start}-{rows_running_end}] cols: [{cols_running_start}-{cols_running_end}] processing started')

    for row_id in range(rows_running_start, rows_running_end):
        print(row_id)
        logger.info(f'tile {tile_name} row {row_id} processing')

        df_matfile = load_cold_reccg(tile_name, row_id, path_cold_reccg)

        if len(df_matfile) == 1:
            logging.info('tile {} row {:04d} no data'.format(tile_name, row_id))
            # print(f'tile {tile_name} row {(row_id):04d} no data')
            continue

        array_t_start = np.array([line[0][0] for line in df_matfile['t_start']])
        array_t_end = np.array([line[0][0] for line in df_matfile['t_end']])
        array_pos = np.array([line[0][0] for line in df_matfile['pos']])

        for col_id in range(cols_running_start, cols_running_end):

            array_is_pct = img_stack_ts_is_pct[:, row_id, col_id]
            array_is_mask = img_stack_ts_is_mask[:, row_id, col_id]

            if (array_is_pct == 255).any():
                print(f'tile {tile_name} row {row_id} col {col_id} has NaN value (255)')
                continue

            # get the mask to locate the position of current pixel in the COLD reccg data
            nrow, ncol = 5000, 5000
            pos = row_id * nrow + col_id + 1

            mask_match = array_pos == pos

            # generate the post-processing mask for each year
            (array_valid_segment_mask, array_seamless_segment_mask) = generate_post_processing_mask(array_t_start,
                                                                                                    array_t_end,
                                                                                                    mask_match,
                                                                                                    array_year,
                                                                                                    )

            # post-processing the ISP time-series data
            array_is_pct_post_processing = post_processing_time_series(array_is_pct,
                                                                       array_is_mask,
                                                                       array_valid_segment_mask,
                                                                       array_seamless_segment_mask,
                                                                       post_processing_rule=post_processing_rule)

            img_stack_ts_is_pct_post_processing[:, row_id, col_id] = array_is_pct_post_processing

    return  img_stack_ts_is_pct_post_processing


def output_post_process_isp_images(img_stack_ts_is_pct_post_processing,
                                   array_year, tile_name,
                                   folder_output,
                                   filename_prefix='unet_regressor_round_masked_post_processing',
                                   rootpath_conus_isp=None):
    """
    save the post-processed ISP images

    :param img_stack_ts_is_pct_post_processing:
    :param array_year:
    :param tile_name:
    :param filename_prefix:
    :param rootpath_conus_isp:
    :return:
    """

    if rootpath_conus_isp is None:
        rootpath_conus_isp = rootpath

    for i_year in range(0, len(array_year)):
        year = array_year[i_year]
        img_isp_post_processing_single_year = img_stack_ts_is_pct_post_processing[i_year, :, :]

        output_path = join(rootpath_conus_isp, 'results', 'conus_isp', folder_output, f'{year}', tile_name)
        if not exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        src_proj, src_geotrans = get_proj_info(tile_name)

        output_filename = predict_isp_output(output_path, tile_name, year,
                                             img_isp_post_processing_single_year, src_geotrans, src_proj,
                                             gdal_type=gdalconst.GDT_Byte,
                                             filename_prefix=filename_prefix)

        add_pyramids_color_in_nlcd_isp_tif(output_filename)



@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID, e.g., 1-32')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
# if __name__ =='__main__':
    # rank = 1
    # n_cores = 2000
    
    path_cold_reccg = r'/shared/zhulab/Shi/ProjectCONUSDisturbanceAgent/Detection/'
    rootpath_conus_isp = r'/shared/zhulab/Falu/CSM_project'

    # path_cold_reccg = join(rootpath, 'data', 'Shi_CONUS_Agent_Detection_record')
    # rootpath_conus_isp = None
    # if rootpath_conus_isp is None:
        # rootpath_conus_isp = rootpath

    array_year = np.arange(1985, 2023, 1)
    rows_running_start = 0
    cols_running_start = 0
    rows_running_end = 5000
    cols_running_end = 5000
    
    post_processing_rule = 'mean'  # max, start, mean, min, median
    folder_output = f'individual_year_tile_post_processing_{post_processing_rule}'
    output_filename_prefix = 'unet_regressor_round_masked_post_processing'

    # df_summary_high_res_isp = pd.read_excel(join(rootpath, 'data', 'ISP_from_high_res_lc', 'summary_high_resolution_isp.xlsx'))
    # list_tile_name, array_tiles_counts = np.unique(df_summary_high_res_isp['tile_id'].values, return_counts=True)
    # list_tile_name = list_tile_name.tolist()
    # list_tile_name = convert_6_tile_names_to_8_tile_names(list_tile_name)
    
    list_conus_tile_name = get_conus_tile_name()
    list_conus_tile_name.sort()
    
    list_tile_name_finished = os.listdir(join(rootpath_conus_isp, 'results', 'conus_isp', folder_output, f'{2022}'))
    mask_running = ~np.isin(list_conus_tile_name, list_tile_name_finished)
    list_tile_running = np.array(list_conus_tile_name)[mask_running].tolist()
    # print(list_tile_running)
    
    each_core_block = int(np.ceil(len(list_tile_running) / n_cores))
    for i in range(0, each_core_block):
        new_rank = rank - 1 + i * n_cores
        if new_rank > len(list_tile_running) - 1:  # means that all folder has been processed
            print(f'{new_rank} this is the last running task')
        else:
            if len(list_tile_running[new_rank]) == 6:
                tile_name = convert_6_tile_names_to_8_tile_names(list_tile_running[new_rank])
            else:
                tile_name = list_tile_running[new_rank]
            print(tile_name)
            # tile_name = 'h027v009'

            output_path_logger = join(rootpath_conus_isp, 'results', 'conus_isp', folder_output, f'{1985}', tile_name)
            if not os.path.exists(output_path_logger):
                os.makedirs(output_path_logger, exist_ok=True)

            logger = define_logger(join(output_path_logger, '{}_isp_post-processing.log'.format(tile_name)))
            logger.info(f'tile {tile_name} post-processing started')
            logger.info(f'tile {tile_name} post-processing rule: {post_processing_rule}')
            logger.info(f'tile {tile_name} post-processing folder: {folder_output}')
            logger.info(f'tile {tile_name} post-processing output filename prefix: {output_filename_prefix}')
            logger.info(f'tile {tile_name} post-processing years: {array_year}')
            logger.info(f'tile {tile_name} post-processing rows: [{rows_running_start}-{rows_running_end}] cols: [{cols_running_start}-{cols_running_end}]')

            img_stack_ts_is_mask = load_conus_isp_stack(list_year=array_year,
                                                tile_name=tile_name,
                                                filename_prefix='unet_classifier',
                                                rootpath_conus_isp=rootpath_conus_isp)

            img_stack_ts_is_pct = load_conus_isp_stack(list_year=array_year,
                                                tile_name=tile_name,
                                                filename_prefix='unet_regressor_round',
                                                rootpath_conus_isp=rootpath_conus_isp)

            logger.info(f'tile {tile_name} load the ISP images completed')

            img_stack_ts_is_pct_post_processing = pipeline_post_processing_isp_ts(img_stack_ts_is_pct=img_stack_ts_is_pct,
                                                                                img_stack_ts_is_mask=img_stack_ts_is_mask,
                                                                                array_year=array_year,
                                                                                path_cold_reccg=path_cold_reccg,
                                                                                post_processing_rule=post_processing_rule,
                                                                                tile_name=tile_name,
                                                                                rows_running_start=rows_running_start,
                                                                                cols_running_start=cols_running_start,
                                                                                rows_running_end=rows_running_end,
                                                                                cols_running_end=cols_running_end,
                                                                                logger=logger)

            # save the post-processing ISP images
            output_post_process_isp_images(img_stack_ts_is_pct_post_processing,
                                        array_year=array_year,
                                        tile_name=tile_name,
                                        folder_output=folder_output,
                                        filename_prefix=output_filename_prefix,
                                        rootpath_conus_isp=rootpath_conus_isp)
            
            # write the finished flag
            # f = open(join(output_path, f'{tile_name}_finished.txt'), 'w')    # type: ignore
            # f.write(f'{tile_name} ISP post-processing running finished')
            # f.close()

            logger.info(f'tile {tile_name} save the post-processed ISP images completed')


if __name__ =='__main__':
    main()

