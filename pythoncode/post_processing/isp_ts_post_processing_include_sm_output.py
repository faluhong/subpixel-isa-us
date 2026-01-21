"""
    post_processing of the ISP time series by combining the IS change and NDVI threshold rule
    Also include the surface modification types in the outputs

    Two major steps
    (1) If IS expansion case, no post_processing is needed.
        For the IS reversal case, it depends on the "surface_modification_flag" variable
    (2) If the ISPs previous and next segments are both larger than 0, then apply the NDVI threshold.
        The NDVI threshold is set as 0.15

        If the absolute NDVI change between these two segments are is less than or equal to the threshold (0.15), then merge the two segments, i.e., regarding no change
        If the absolute NDVI change between these two segments are is larger than the threshold (0.15), then no merge, i.e., regarding as change

    The output include the post-processed ISP images and IS change types including the surface modification types

    The surface modification types does not change the ISP values, which cannot be obtained from the ISP time series changes,
    but surface modification is valuable, so it is included in the IS change type images
    
    For the running time, when using 110 cores, it takes about 24 hours for the whole CONUS (427 tile)
"""

import numpy as np
from os.path import join, exists
import click
import logging
import os
import sys
from osgeo import gdal, gdalconst

pwd = os.getcwd()
rootpath = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from util_function.utils_cal_index import (ndvi_cal)
from pythoncode.util_function.tile_name_convert import convert_6_tile_names_to_8_tile_names

from pythoncode.conus_isa_production.utils_get_conus_tile_name import get_conus_tile_name
from pythoncode.model_training.utils_deep_learning import (add_pyramids_color_in_is_change_type_tif, get_proj_info)

from pythoncode.post_processing.isp_ts_post_processing import (define_logger,
                                                               load_conus_isp_stack,
                                                               load_cold_reccg,
                                                               generate_post_processing_mask,
                                                               post_processing_time_series,
                                                               output_post_process_isp_images,
                                                               find_modes)

from pythoncode.post_processing.utils_apply_permanent_natural_surface_mask import (apply_permanent_natural_surface_us_land_boundary_mask)


def binary_is_ndvi_based_post_processing_with_sm(array_valid_segment_mask_ori,
                                                 array_seamless_segment_mask_ori,
                                                 array_ndvi_end,
                                                 array_ndvi_begin,
                                                 array_is_mask,
                                                 mask_match,
                                                 threshold=0.15,
                                                 surface_modification_flag='binary_is_ndvi'):
    """
    refine the segment mask by combining the binary IS change (IS expansion and reversal) and the NDVI threshold

    (1) If IS expansion case, i.e., previous segment is natural surface (ISP = 0) and the ISP of next segment is larger than 0,
        or IS reversal case, i.e., the previous segment is impervious surface (ISP > 1) and the ISP of next segment is 0 (natural surface),
        no post_processing is needed.
    (2) If ISP change case (IS intensification and decline), applying the NDVI threshold

    If the NDVI change between the end of the previous segment and the beginning of the next segment is smaller than the threshold,
    merge the two segments

    Args:
        array_valid_segment_mask_ori (np.array): the segment mask
        array_seamless_segment_mask_ori (np.array): the seamless segment mask
        array_ndvi_end (np.array): the NDVI at the end of the segment
        array_ndvi_begin (np.array): the NDVI at the beginning of the segment
        array_is_mask (np.array): array to record the IS / Natural surface mask for each year
        mask_match (np.array): the mask to locate the position of current pixel in the COLD reccg data
        threshold (float): the NDVI threshold, default is 0.15

    Returns:
        array_valid_segment_mask_update: the updated mask to indicate which years contain valid COLD segment for the post_processing
        array_seamless_segment_mask_update: the updated seamless mask for the post_processing
        array_surface_modificaton_mask: the mask to indicate which year-to-year change is
    """

    array_valid_segment_mask_update = array_valid_segment_mask_ori.copy()
    array_seamless_segment_mask_update = array_seamless_segment_mask_ori.copy()

    array_surface_modification_mask = np.zeros((len(array_seamless_segment_mask_ori) - 1), dtype=bool)

    # list_segment_id_valid = np.unique(array_valid_segment_mask_ori)
    list_segment_id_seamless = np.unique(array_seamless_segment_mask_ori)

    if len(list_segment_id_seamless) == 1:
        # only one segment, no need to use the NDVI threshold
        pass
    else:
        # multiple segments, use the NDVI threshold to merge the segments where the NDVI change is smaller than the threshold

        # get the NDVI values of the end of the segment and the beginning of the next segment
        array_ndvi_end_match = array_ndvi_end[mask_match]
        array_ndvi_begin_match = array_ndvi_begin[mask_match]

        # for loop from the second segment to the last segment to compare the NDVI change
        for i_seg in range(1, len(list_segment_id_seamless)):

            seg_id = list_segment_id_seamless[i_seg]

            # check the condition of previous segment
            mask_seg_valid_previous = array_valid_segment_mask_ori == seg_id - 1  # the valid segment mask, used for determining the segment

            if np.sum(mask_seg_valid_previous) == 0:
                # if there is no valid segment, then apply the seamless segment mask
                # the case is a special case, due to the first segment is before 1985-7-1 and the target year is 1985
                # in this case, array_valid_segment_mask_ori: 0 2 2 2; array_seamless_segment_mask_ori: 1 2 2 2
                # we cannot find the previous segment ID from the valid segment mask, so we need to use the seamless segment mask
                # print(array_valid_segment_mask_ori, array_seamless_segment_mask_ori, seg_id)
                mask_seg_valid_previous = array_seamless_segment_mask_ori == seg_id - 1

            # check the condition of current segment
            mask_seg_valid_current = array_valid_segment_mask_ori == seg_id  # the valid segment mask, used for determining the segment
            if np.sum(mask_seg_valid_current) == 0:
                mask_seg_valid_current = array_seamless_segment_mask_ori == seg_id

            # using the majority vote to determine the pre-segment is natural surface or impervious surface
            is_mask_seg_previous = array_is_mask[mask_seg_valid_previous]
            is_mask_seg_modes_previous = find_modes(is_mask_seg_previous)  # the modes of the segment

            is_mask_seg_current = array_is_mask[mask_seg_valid_current]
            is_mask_seg_modes_current = find_modes(is_mask_seg_current)  # the modes of the segment

            assert len(is_mask_seg_modes_previous) < 3, 'The pre-segment contain >=3 modes, please check the data'
            assert len(is_mask_seg_modes_current) < 3, 'The current segment contain >=3 modes, please check the data'

            # determine the pre-segment and current segment is natural surface or impervious surface
            # if contains two modes, it means the natural surface and impervious surface are equal, then regard as natural surface
            # if the first mode is 0, it means the natural surface is dominant, then regard as natural surface
            # true means natural surface, false means impervious surface
            flag_natural_previous = (len(is_mask_seg_modes_previous) == 2) | (is_mask_seg_modes_previous[0] == 0)
            flag_natural_current = (len(is_mask_seg_modes_current) == 2) | (is_mask_seg_modes_current[0] == 0)

            if (flag_natural_previous | flag_natural_current) & (surface_modification_flag == 'binary_is_ndvi'):
                # If any one of the previous or current segment is natural surface, then no NDVI post_processing is needed
                # It contains two types: IS expansion and IS reversal
                # no NDVI post_processing is needed
                pass
            elif (flag_natural_previous ) & (surface_modification_flag == 'is_expansion_ndvi'):
                # if previous segment is natural surface, then no NDVI post_processing is needed
                # Even though the current segment is natural surface, NDVI post_processing is needed
                # The post_processing will be applied to IS reversal types, but not IS expansion types
                pass
            else:

                if np.abs(array_ndvi_end_match[i_seg - 1] - array_ndvi_begin_match[i_seg]) <= threshold:
                    # the NDVI change is smaller than the threshold, merge the two segments

                    # get the mask of the previous segment, use the original seamless segment array to get the mask
                    mask_previous_segment = array_seamless_segment_mask_ori == list_segment_id_seamless[i_seg - 1]
                    assert np.count_nonzero(mask_previous_segment) > 0, 'cannot find the previous segment'

                    # get the previous segment ID, use the update segment array to get the previous segment ID
                    # because the previous segment ID has been updated, for example original segment is 1->2->3, after first update, it is 1->1->3.
                    # If the third segment also need to be updated, the previous segment ID should be 1, not 2.

                    # get the previous segment ID, use the updated seamless segment array to get the previous segment ID
                    array_pre_segment_ids = array_seamless_segment_mask_update[mask_previous_segment].copy()
                    array_pre_segment_ids = array_pre_segment_ids[array_pre_segment_ids != 0]  # ignore the zero values
                    assert len(np.unique(array_pre_segment_ids)) == 1, 'The previous segment ID is not unique, please check the data'

                    previous_seg_id = array_pre_segment_ids[0]

                    # change the current segment ID to the previous segment ID
                    array_valid_segment_mask_update[array_valid_segment_mask_update == list_segment_id_seamless[i_seg]] = previous_seg_id
                    array_seamless_segment_mask_update[array_seamless_segment_mask_update == list_segment_id_seamless[i_seg]] = previous_seg_id

                    # assign the surface modification label
                    array_surface_modification_mask[np.where(mask_previous_segment)[0][-1]] = True

                    # print(i_seg, np.abs(array_ndvi_end_match[i_seg - 1] - array_ndvi_begin_match[i_seg]) <= threshold,
                    #       array_seamless_segment_mask_update)

    return (array_valid_segment_mask_update, array_seamless_segment_mask_update, array_surface_modification_mask)


def get_is_change_type_time_series(array_is_pct_post_processing_update, array_surface_modification_mask):
    """
        get the IS change time series based on the update post-processed ISP time series

        :param array_is_pct_post_processing_update: the ISP time series
        :param array_surface_modification_mask: the mask to indicate the surface modification
        :return:
    """

    array_is_change_types = np.zeros_like(array_surface_modification_mask, dtype=np.int8)

    for p in range(0, len(array_is_pct_post_processing_update) - 1):

        isp_year_1 = array_is_pct_post_processing_update[p]
        isp_year_2 = array_is_pct_post_processing_update[p + 1]

        if (isp_year_1 == 0) & (isp_year_2 == 0):
            # stable natural surface
            array_is_change_types[p] = 1
        elif (isp_year_1 > 0) & (isp_year_2 > 0) & (isp_year_1 == isp_year_2):
            # stable IS
            array_is_change_types[p] = 2
        elif (isp_year_1 == 0) & (isp_year_2 > 0):
            # IS expansion
            array_is_change_types[p] = 3
        elif (isp_year_1 > 0) & (isp_year_2 > 0) & (isp_year_1 < isp_year_2):
            # IS intensification
            array_is_change_types[p] = 4
        elif (isp_year_1 > 0) & (isp_year_2 > 0) & (isp_year_1 > isp_year_2):
            # IS decline
            array_is_change_types[p] = 5
        elif (isp_year_1 > 0) & (isp_year_2 == 0):
            # IS reversal
            array_is_change_types[p] = 6

    # assign the surface modification label
    array_is_change_types[array_surface_modification_mask] = 7

    return array_is_change_types


def pipeline_post_processing_isp_ts_with_sm_labels(img_stack_ts_is_pct,
                                                   img_stack_ts_is_mask,
                                                   array_year, path_cold_reccg,
                                                   post_processing_rule,
                                                   tile_name, rows_running_start, cols_running_start,
                                                   rows_running_end, cols_running_end,
                                                   logger,
                                                   threshold,
                                                   surface_modification_flag):
    """
    pipeline for post_processing the ISP time series by considering the type of IS change and the NDVI threshold

    More specifically:
        (1) Apply the NDVI threshold to IS intensification and decline types
        (2) No post_processing on the IS expansion types
        (3) The post_processing on the IS reversal types depends on the "surface_modification_flag" flag
            if  "surface_modification_flag" is "binary_is_ndvi", no post_processing on the IS reversal types, use binary_is_ndvi_based_post_processing function
            if  "surface_modification_flag" is "is_expansion_ndvi", apply the NDVI threshold to the IS reversal types, use is_expansion_ndvi_based_post_processing function

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

    # define the array of IS change types, the shape is (n_year - 1, n_row, n_col). It contains the year-to-year IS change types
    img_stack_is_change_types = np.zeros((np.shape(img_stack_ts_is_pct)[0] - 1,
                                          np.shape(img_stack_ts_is_pct)[1],
                                          np.shape(img_stack_ts_is_pct)[2]), dtype=np.uint8)

    logger.info(f'tile: {tile_name} rows: [{rows_running_start}-{rows_running_end}] cols: [{cols_running_start}-{cols_running_end}] processing started')

    for row_id in range(rows_running_start, rows_running_end):
        print(f'row id: {row_id}')

        logger.info(f'tile {tile_name} row {row_id} processing')

        df_matfile = load_cold_reccg(tile_name, row_id, path_cold_reccg)

        if len(df_matfile) == 1:
            logging.info('tile {} row {:04d} no data'.format(tile_name, row_id))
            # print(f'tile {tile_name} row {(row_id):04d} no data')
            continue

        # get the COLD reccg data and calculate the NDVI
        array_t_start = np.array([line[0][0] for line in df_matfile['t_start']])
        array_t_end = np.array([line[0][0] for line in df_matfile['t_end']])
        array_pos = np.array([line[0][0] for line in df_matfile['pos']])

        array_a0 = np.array([line[0, :] for line in df_matfile['coefs']])
        array_c1 = np.array([line[1, :] for line in df_matfile['coefs']])

        # the overall SR at the start and the end of the segment
        array_overall_sr_start = array_a0 + array_c1 * (np.tile(array_t_start, (7, 1)).T)
        array_overall_sr_end = array_a0 + array_c1 * (np.tile(array_t_end, (7, 1)).T)

        array_red_begin = array_overall_sr_start[:, 2]
        array_nir_begin = array_overall_sr_start[:, 3]

        array_red_end = array_overall_sr_end[:, 2]
        array_nir_end = array_overall_sr_end[:, 3]

        array_ndvi_begin = ndvi_cal(array_red_begin, array_nir_begin)
        array_ndvi_end = ndvi_cal(array_red_end, array_nir_end)

        for col_id in range(cols_running_start, cols_running_end):
            # print(f'column id: {col_id}')

            array_is_pct = img_stack_ts_is_pct[:, row_id, col_id]
            array_is_mask = img_stack_ts_is_mask[:, row_id, col_id]

            if (array_is_pct == 255).any():
                print(f'tile {tile_name} row {row_id} col {col_id} has NaN value (255)')
                continue

            # get the mask to locate the position of current pixel in the COLD reccg data
            nrow, ncol = 5000, 5000
            pos = row_id * nrow + col_id + 1

            mask_match = array_pos == pos

            # generate the post_processing mask for each year
            (array_valid_segment_mask_ori, array_seamless_segment_mask_ori) = generate_post_processing_mask(array_t_start,
                                                                                                            array_t_end,
                                                                                                            mask_match,
                                                                                                            array_year,)

            # further refine the segment mask using the NDVI threshold
            (array_valid_segment_mask_update,
            array_seamless_segment_mask_update,
            array_surface_modification_mask) = binary_is_ndvi_based_post_processing_with_sm(array_valid_segment_mask_ori,
                                                                                            array_seamless_segment_mask_ori,
                                                                                            array_ndvi_end,
                                                                                            array_ndvi_begin,
                                                                                            array_is_mask,
                                                                                            mask_match,
                                                                                            threshold=threshold,
                                                                                            surface_modification_flag=surface_modification_flag)
        
            # post_processing the ISP time-series data
            array_is_pct_post_processing_update = post_processing_time_series(array_is_pct,
                                                                              array_is_mask,
                                                                              array_valid_segment_mask_update,
                                                                              array_seamless_segment_mask_update,
                                                                              post_processing_rule=post_processing_rule)

            array_is_change_types = get_is_change_type_time_series(array_is_pct_post_processing_update,
                                                                   array_surface_modification_mask)

            # assign the post-processed ISP time series and the IS change types to the output array
            img_stack_is_change_types[:, row_id, col_id] = array_is_change_types
            img_stack_ts_is_pct_post_processing[:, row_id, col_id] = array_is_pct_post_processing_update

    return  (img_stack_ts_is_pct_post_processing, img_stack_is_change_types)


def output_is_change_type(output_filename, tile_name, img_urban_type, gdal_type=gdalconst.GDT_Byte):
    """
    output the is change type

    typical IS change types include:
    dict_is_change_type = {'1': 'stable natural',
                               '2': 'stable IS',
                               '3': 'IS expansion',
                               '4': 'IS intensification',
                               '5': 'IS decline',
                               '6': 'IS reversal',
                               '7': 'surface modification'}

    :param output_path:
    :param tile_name:
    :param year:
    :param img_urban_type:
    :param src_geotrans:
    :param src_proj:
    :return:
    """

    src_proj, src_geotrans = get_proj_info(tile_name)

    if not exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    n_row, n_col = img_urban_type.shape[1], img_urban_type.shape[0]
    tif_output = gdal.GetDriverByName('GTiff').Create(output_filename, n_col, n_row, 1, gdal_type, options=['COMPRESS=LZW'])
    tif_output.SetGeoTransform(src_geotrans)
    tif_output.SetProjection(src_proj)

    Band = tif_output.GetRasterBand(1)
    Band.WriteArray(img_urban_type)

    tif_output = None
    del tif_output

    return output_filename


def output_post_process_is_change_type_images(img_stack_is_change_types,
                                              array_year,
                                              tile_name,
                                              folder_output,
                                              rootpath_conus_isp=None):
    """
    save the post-processed IS change types

    :param img_stack_is_change_types:
    :param array_year:
    :param tile_name:
    :param filename_prefix:
    :param rootpath_conus_isp:
    :return:
    """


    if rootpath_conus_isp is None:
        rootpath_conus_isp = rootpath

    colors = np.array([np.array([108, 169, 102, 255]) / 255,  # stable natural
                       np.array([179, 175, 164, 255]) / 255,  # stable IS
                       np.array([255, 0, 0, 255]) / 255,  # IS expansion
                       np.array([126, 30, 156, 255]) / 255,  # IS intensification
                       np.array([250, 192, 144, 255]) / 255,  # IS decline
                       np.array([29, 101, 51, 255]) / 255,  # IS reversal
                       np.array([130, 201, 251, 255]) / 255,  # Surface modification
                       ])

    for i_year in range(0, len(array_year) - 1):
        year = array_year[i_year]
        img_isp_post_processing_single_year = img_stack_is_change_types[i_year, :, :]

        output_path = join(rootpath_conus_isp, 'results', 'conus_isp', folder_output, f'{year}-{year + 1}', tile_name)
        if not exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        output_filename = join(output_path, '{}_{}_{}_{}_is_change_type.tif'.format(tile_name, folder_output, year, year + 1))

        output_filename_urban_type = output_is_change_type(output_filename, tile_name, img_isp_post_processing_single_year,
                                                           gdal_type=gdalconst.GDT_Byte)

        add_pyramids_color_in_is_change_type_tif(output_filename_urban_type, list_overview=None, colors=colors)


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

    post_processing_rule = 'mean'  # max, start, mean

    vi_post_processing_rule = 'binary_is_ndvi015_sm'
    threshold = 0.15
    surface_modification_flag = 'binary_is_ndvi'

    folder_output = f'individual_year_tile_post_processing_{vi_post_processing_rule}'
    output_filename_prefix = 'unet_regressor_round_masked_post_processing'
    
    list_conus_tile_name = get_conus_tile_name()
    list_conus_tile_name.sort()
    
    if not exists(join(rootpath_conus_isp, 'results', 'conus_isp', folder_output, f'{2022}')):
        list_tile_name_finished = []
    else:
        list_tile_name_finished = os.listdir(join(rootpath_conus_isp, 'results', 'conus_isp', folder_output, f'{2022}'))
        
    mask_running = ~np.isin(list_conus_tile_name, list_tile_name_finished)
    list_tile_running = np.array(list_conus_tile_name)[mask_running].tolist()
    print(list_tile_running)

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

            output_path_logger = join(rootpath_conus_isp, 'results', 'conus_isp', folder_output, f'{1985}', tile_name)
            if not os.path.exists(output_path_logger):
                os.makedirs(output_path_logger, exist_ok=True)

            logger = define_logger(join(output_path_logger, '{}_isp_post-processing.log'.format(tile_name)))
            logger.info(f'tile {tile_name} post_processing started')
            logger.info(f'tile {tile_name} post_processing rule: {post_processing_rule}')
            logger.info(f'tile {tile_name} post_processing folder: {folder_output}')
            logger.info(f'tile {tile_name} post_processing output filename prefix: {output_filename_prefix}')
            logger.info(f'tile {tile_name} post_processing years: {array_year}')
            logger.info(f'tile {tile_name} post_processing rows: [{rows_running_start}-{rows_running_end}] cols: [{cols_running_start}-{cols_running_end}]')
            logger.info(f'tile {tile_name} post_processing surface modification flag: {surface_modification_flag}')

            img_stack_ts_is_mask = load_conus_isp_stack(list_year=array_year,
                                                        tile_name=tile_name,
                                                        filename_prefix='unet_classifier',
                                                        rootpath_conus_isp=rootpath_conus_isp)

            img_stack_ts_is_pct = load_conus_isp_stack(list_year=array_year,
                                                       tile_name=tile_name,
                                                       filename_prefix='unet_regressor_round',
                                                       rootpath_conus_isp=rootpath_conus_isp)

            logger.info(f'tile {tile_name} load the ISP images completed')

            (img_stack_ts_is_pct_post_processing,
            img_stack_is_change_types) = pipeline_post_processing_isp_ts_with_sm_labels(img_stack_ts_is_pct=img_stack_ts_is_pct,
                                                                                        img_stack_ts_is_mask=img_stack_ts_is_mask,
                                                                                        array_year=array_year,
                                                                                        path_cold_reccg=path_cold_reccg,
                                                                                        post_processing_rule=post_processing_rule,
                                                                                        tile_name=tile_name,
                                                                                        rows_running_start=rows_running_start,
                                                                                        cols_running_start=cols_running_start,
                                                                                        rows_running_end=rows_running_end,
                                                                                        cols_running_end=cols_running_end,
                                                                                        logger=logger,
                                                                                        threshold=threshold,
                                                                                        surface_modification_flag=surface_modification_flag)

            img_stack_ts_is_pct_post_processing = apply_permanent_natural_surface_us_land_boundary_mask(img_stack_ts_is_pct_post_processing,
                                                                                                        tile_name,
                                                                                                        label_permanent_natural_surface=0)

            img_stack_is_change_types = apply_permanent_natural_surface_us_land_boundary_mask(img_stack_is_change_types,
                                                                                              tile_name,
                                                                                              label_permanent_natural_surface=1)

            # save the post_processing ISP images
            output_post_process_isp_images(img_stack_ts_is_pct_post_processing,
                                        array_year=array_year,
                                        tile_name=tile_name,
                                        folder_output=folder_output,
                                        filename_prefix=output_filename_prefix,
                                        rootpath_conus_isp=rootpath_conus_isp)

            # save the post_processing IS change types images
            output_post_process_is_change_type_images(img_stack_is_change_types=img_stack_is_change_types,
                                                      array_year=array_year,
                                                      tile_name=tile_name,
                                                      folder_output=folder_output,
                                                      rootpath_conus_isp=rootpath_conus_isp)

            logger.info(f'tile {tile_name} save the post-processed ISP images completed')


if __name__ =='__main__':
    main()


