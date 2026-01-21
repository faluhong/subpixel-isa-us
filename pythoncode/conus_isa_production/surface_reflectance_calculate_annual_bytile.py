"""
    Extract variables annually from COLD rec_cg files provided by Shi Qiu
    (1) overall surface reflectance, include two parts:
        (1.1) central overall surface reflectance

            overall_surface_reflectance = array_a0 + array_c1 * 0.5 * (array_t_start_segment, array_t_end_segment)
            array_t_start_segment: start date number of the segment
            array_t_end_segment: end date number of the segment

        (1.2) annual overall surface reflectance

            overall_surface_reflectance = array_a0 + array_c1 * 0.5 * (array_t_start_each_year, array_t_end_each_year)

            array_t_start_each_year: start date number of each year
            array_t_end_each_year: end date number of each year

    (2) Fitting RMSE
    (3) Harmonic coefficients

    The codes contain the basic functions to extract the temporal features from the COLD rec_cg files

    Notes after testing:
    (1) The I/O is intensive, so better run it on shared/zhulab or scratch folder.
        Running on cn450 fully occupied the I/O and takes a long time to save the outputs
    (2) There is a trade-off between the memory usage and the speed of the extraction
        The more target years (defined by list_year_for_extraction), the more memory usage but the faster speed
        For the 38 years (1985-2022), the total memory is more than 250 GB, which is not recommended
        For 5 years running, finishing one tiles takes about 2 days with memory usage of more than 36 GB
        Considering the speed, running annually or bi-annually is the good way to go.
    (3) Storing the 38 year (1985-2022) predictor variables for one tile takes on average 200 GBs (180-220 GBs).
        The predictor variables include the
            stable surface reflectance,
            annual surface reflectance,
            RMSE,
            c1,
            a1, b1, a2, b2, a3, b3
        In total, 70 variables (10 multiple 7 bands).
"""

import numpy as np
import os
from os.path import join
import sys
import scipy.io as scio
from osgeo import gdal, gdalconst
from datetime import datetime
import logging

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, "../.."))
path_basictools = os.path.join(rootpath_project, 'pythoncode')
sys.path.append(path_basictools)

from pythoncode.util_function.datetime_datenum_convert import datetime_to_datenum_matlabversion
from util_function.mat_to_dataframe import mat_to_dataframe
from pythoncode.model_training.utils_deep_learning import get_proj_info


def get_temporal_feature_image_tile_year(path_cold, tile_name, list_year_for_extraction, nan_fill=0, rows_running=5000):
    """
        get the surface reflectance image from the COLD rec_cg file
    Args:
        path_cold: the path of the COLD rec_cg file
        tile_name: e.g., h026v006
        list_year_for_extraction: the list of years for extraction, such as np.arange(1985, 2022)
        nan_fill: the value used to fill the nan value when the final image contains nan value, default is 0
    Returns:
        img_sr_stable: the overall reflectance extracted from the segment, i.e., using the central overall reflectance within the segment
        img_sr_change: the overall reflectance extracted annually
        img_rmse: the fitting RMSE of each band
        img_harmonic_coef: the harmonic coefficients of each band, currently, 6 harmonic coefficients (i.e., 3 order) for each band
    """

    path_record_change = join(path_cold, tile_name, 'TSFitLine')

    NRows, NCols = 5000, 5000

    # define the output image to store the extracted variables
    img_sr_stable = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0
    img_sr_change = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0

    img_c1 = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0

    img_rmse = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0

    img_a1 = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0
    img_b1 = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0
    img_a2 = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0
    img_b2 = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0
    img_a3 = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0
    img_b3 = np.zeros((len(list_year_for_extraction), 7, NRows, NCols), dtype=np.float32) + 9999.0

    # img_harmonic_coef = np.zeros((len(list_year_for_extraction), 42, NRows, NCols), dtype=np.float32) + 9999.0

    for row_id in range(0, rows_running):

        if row_id % 1000 == 0:
            logging.info('tile {} row {:04d} extraction'.format(tile_name, row_id))

        file_name_ydata_training = join(path_record_change, 'record_change_r0{:04d}.mat'.format(row_id + 1))

        data = scio.loadmat(file_name_ydata_training, verify_compressed_data_integrity=False)
        mat_rec_cg = data['rec_cg']
        df_matfile = mat_to_dataframe(mat_rec_cg)

        if len(df_matfile) == 1:
            logging.info('tile {} row {:04d} no data'.format(tile_name, row_id))
            # print('tile {} row {:04d} no data'.format(tile_name, row_id))
            continue

        (array_t_start, array_t_end, array_pos, array_rmse,
         array_central_overall_sr, array_c1,
         array_a1, array_b1,
         array_a2, array_b2,
         array_a3, array_b3) = get_variable_from_cold_reccg_stable(df_matfile)

        for i_year in range(0, len(list_year_for_extraction)):
            year = list_year_for_extraction[i_year]

            datetime_target = datetime(year=year, month=7, day=1)
            datenum_target = datetime_to_datenum_matlabversion(datetime_target)

            # generate the mask for matching the COLD parameters with the target date
            mask_same_period, series_pos = mask_flag_generate(datenum_target, datetime_target, array_t_start, array_t_end, array_pos, row_id)

            array_annual_overall_sr = get_annual_overall_reflectance(df_matfile, datenum_target)

            img_sr_stable[i_year, :, row_id, series_pos] = array_central_overall_sr[mask_same_period, :]
            img_sr_change[i_year, :, row_id, series_pos] = array_annual_overall_sr[mask_same_period, :]

            img_c1[i_year, :, row_id, series_pos] = array_c1[mask_same_period, :]

            img_rmse[i_year, :, row_id, series_pos] = array_rmse[mask_same_period, :]

            img_a1[i_year, :, row_id, series_pos] = array_a1[mask_same_period, :]
            img_b1[i_year, :, row_id, series_pos] = array_b1[mask_same_period, :]
            img_a2[i_year, :, row_id, series_pos] = array_a2[mask_same_period, :]
            img_b2[i_year, :, row_id, series_pos] = array_b2[mask_same_period, :]
            img_a3[i_year, :, row_id, series_pos] = array_a3[mask_same_period, :]
            img_b3[i_year, :, row_id, series_pos] = array_b3[mask_same_period, :]

            # img_harmonic_coef[i_year, :, row_id, series_pos] = array_intra_annual_coefs[mask_same_period, :]

    img_sr_stable[img_sr_stable == 9999.0] = nan_fill
    img_sr_change[img_sr_change == 9999.0] = nan_fill

    img_c1[img_c1 == 9999.0] = nan_fill

    img_rmse[img_rmse == 9999.0] = nan_fill
    # img_harmonic_coef[img_harmonic_coef == 9999.0] = nan_fill

    img_a1[img_a1 == 9999.0] = nan_fill
    img_b1[img_b1 == 9999.0] = nan_fill
    img_a2[img_a2 == 9999.0] = nan_fill
    img_b2[img_b2 == 9999.0] = nan_fill
    img_a3[img_a3 == 9999.0] = nan_fill
    img_b3[img_b3 == 9999.0] = nan_fill

    return img_sr_stable, img_sr_change, img_rmse, img_c1, img_a1, img_b1, img_a2, img_b2, img_a3, img_b3


def get_variable_from_cold_reccg_stable(df_matfile):
    """
        get the variable from cold_reccg

        :param
            df_matfile: the dataframe get from the COLD reccg .mat file
        :return:

    """
    length_rec_cg = len(df_matfile)

    # get the central overall reflectance
    array_t_start = np.array([line[0][0] for line in df_matfile['t_start']])
    array_t_end = np.array([line[0][0] for line in df_matfile['t_end']])

    array_pos = np.array([line[0][0] for line in df_matfile['pos']])

    # get the RMSE
    array_rmse = np.array([line for line in df_matfile['rmse']])[:, :, 0]

    # get the intra-annual coefficients, i.e., the harmonic coefficients
    # The shape of the 'coefs' is (8, 7), where the first two rows are the overall reflectance coefficients, and the rest are the harmonic coefficients
    # 7 represents the number of bands
    # array_intra_annual_coefs = np.array([line[2::, :] for line in df_matfile['coefs']]).reshape((length_rec_cg, 42), order='F')

    array_a0 = np.array([line[0, :] for line in df_matfile['coefs']])
    array_c1 = np.array([line[1, :] for line in df_matfile['coefs']])

    array_a1 = np.array([line[2, :] for line in df_matfile['coefs']])
    array_b1 = np.array([line[3, :] for line in df_matfile['coefs']])
    array_a2 = np.array([line[4, :] for line in df_matfile['coefs']])
    array_b2 = np.array([line[5, :] for line in df_matfile['coefs']])
    array_a3 = np.array([line[6, :] for line in df_matfile['coefs']])
    array_b3 = np.array([line[7, :] for line in df_matfile['coefs']])

    array_central_overall_sr = array_a0 + array_c1 * 0.5 * (np.tile(array_t_start, (7, 1)).T + np.tile(array_t_end, (7, 1)).T)

    return (array_t_start, array_t_end, array_pos, array_rmse, array_central_overall_sr, array_c1, array_a1, array_b1, array_a2, array_b2, array_a3, array_b3)


def get_annual_overall_reflectance(df_matfile, datenum_target):
    """
        get the overall reflectance based on the target date number
    :param df_matfile:
    :param datenum_target:
    :return:
    """

    array_a0 = np.array([line[0, :] for line in df_matfile['coefs']])
    array_c1 = np.array([line[1, :] for line in df_matfile['coefs']])

    array_annual_overall_sr = array_a0 + array_c1 * datenum_target

    return array_annual_overall_sr


def mask_flag_generate(datenum_target, datetime_target, array_t_start, array_t_end, array_pos, row_id, NCols=5000):
    """
        generate the mask for matching the COLD parameters with the target date, include those following steps:
        (1) the target date locates within the segment
        (2) the target date locates between two segment, use the latter segment
        (3) the target date locates at outside the beginning/end segment, use the first/last segment
        (4) No segment exists for the pixel, means no COLD-fitted results

        Using the previous and next segments fill in those values is helpful to make the time-series consistent
    """

    # mask for the target date locates within the segment
    mask_same_period = (array_t_start <= datenum_target) & (datenum_target <= array_t_end)

    # get the position that already have the matched segment
    series_pos = array_pos[mask_same_period] - row_id * NCols - 1

    # for loop to find the remaining matched segments
    for p in range(0, NCols):
        if p not in series_pos:

            list_location = np.where(array_pos == p + 1 + NCols * row_id)[0]

            if len(list_location) == 0:
                # if true, means there is no COLD running results
                pass
            elif len(list_location) == 1:
                # if true, means only one fitted segment, then use the only one
                mask_same_period[list_location] = True
            else:
                match_flag = 0
                for loc_idx in range(0, len(list_location) - 1):
                    # if the target datetime is between two segments, segments closer to the target date is used
                    if (array_t_end[list_location[loc_idx]] < datenum_target) & (array_t_start[list_location[loc_idx] + 1] > datenum_target):

                        # calculate the distance to the previous and next segments
                        distance_previous_segment = np.abs(datenum_target - array_t_end[list_location[loc_idx]])
                        distance_next_segment = np.abs(array_t_start[list_location[loc_idx] + 1] - datenum_target)

                        if distance_previous_segment >= distance_next_segment:
                            # next segment is selected due to closer distance
                            mask_same_period[list_location[loc_idx + 1]] = True
                        else:
                            mask_same_period[list_location[loc_idx]] = True
                        match_flag = 1

                if match_flag == 0:
                    # if the target datetime is not between two segment, the target datetime could be at the
                    # beginning or the end of the whole time series, use 2000 as the threshold, if target year is in
                    # or before 2000, use the first segment, else, use the last segment
                    if datetime_target.year <= 2000:
                        mask_same_period[list_location[0]] = True
                    else:
                        mask_same_period[list_location[loc_idx] + 1] = True

    # update the series_pos after
    series_pos_update = array_pos[mask_same_period] - row_id * NCols - 1

    return mask_same_period, series_pos_update


def output_temporal_feature_image(surface_ref_image, tile_name, year, output_bandname, src_geotrans, src_proj, output_path):
    """
    output the surface reflectance image

    Args:
        surface_ref_image
        tile_name: e.g., h026v006
        year: e.g., 2001
        output_bandname:
        src_geotrans:
        src_proj:
        path_data:
    Returns:
        None
    """

    if len(surface_ref_image.shape) == 2:
        layer_number = 1
        n_rows, n_cols = np.shape(surface_ref_image)[0], np.shape(surface_ref_image)[1]
    else:
        layer_number = np.shape(surface_ref_image)[0]
        n_rows, n_cols = np.shape(surface_ref_image)[1], np.shape(surface_ref_image)[2]

    output_filename = join(output_path, '{}_{}_{}.tif'.format(tile_name, output_bandname, year))

    tif_output = gdal.GetDriverByName('GTiff').Create(output_filename, n_cols, n_rows, layer_number, gdalconst.GDT_Float32, options=['COMPRESS=LZW'])
    tif_output.SetGeoTransform(src_geotrans)
    tif_output.SetProjection(src_proj)

    if layer_number == 1:
        band = tif_output.GetRasterBand(1)
        band.WriteArray(surface_ref_image)
    else:
        for i_layer in range(0, layer_number):
            band = tif_output.GetRasterBand(i_layer + 1)
            band.WriteArray(surface_ref_image[i_layer, :, :])

    del tif_output

    return None


def pipeline_extract_temporal_feature(path_cold, tile_name, list_year_extraction, nan_fill=0, rows_running=5000,
                                      output_project_folder=None,
                                      sr_stable_output_flag=True,
                                      predictor_variable_folder='predictor_variable'):
    """
        pipeline to extract the predictor variables from the COLD rec_cg files
        (1) extract the images of the temporal features
        (2) output the images

        :param path_cold: the path of the COLD rec_cg file
        :param tile_name: tile_name: e.g., h026v006
        :param list_year_extraction: the list of years for extraction, such as np.arange(1985, 2022)
        :param nan_fill: the value used to fill the nan value when the final image contains nan value, default is 0
        :param rows_running: the number of rows running
        :param output_project_folder: the output folder to store the extracted variables, default is None, then the output folder is the rootpath_project
        :param sr_stable_output_flag: the flag to output the stable (central) surface reflectance, default is True
        :param predictor_variable_folder: the folder to store the predictor variables, default is 'predictor_variable'

        :return:
    """

    # load projection information
    src_proj, src_geotrans = get_proj_info(tile_name)

    logging.info('tile {} start: extract predictor variables from rec_cg'.format(tile_name))
    logging.info('extraction period: {}'.format(list_year_extraction))

    (img_sr_stable, img_sr_change,
     img_rmse, img_c1,
     img_a1, img_b1,
     img_a2, img_b2,
     img_a3, img_b3) = get_temporal_feature_image_tile_year(path_cold, tile_name,
                                                            list_year_for_extraction=list_year_extraction,
                                                            nan_fill=nan_fill,
                                                            rows_running=rows_running)

    logging.info('tile {} done: extract predictor variables from rec_cg'.format(tile_name))

    for i_year in range(0, len(list_year_extraction)):

        year = list_year_extraction[i_year]

        if output_project_folder is not None:
            output_path_root = join(output_project_folder, 'data', predictor_variable_folder, tile_name, str(year))
        else:
            # if not specified, the output folder is the rootpath_project
            output_path_root = join(rootpath_project, 'data', predictor_variable_folder, tile_name, str(year))

        if not os.path.exists(output_path_root):
            os.makedirs(output_path_root, exist_ok=True)

        # output extracted variables
        logging.info('tile {}  year {}: output variables'.format(tile_name, year))

        if sr_stable_output_flag is True:
            output_temporal_feature_image(img_sr_stable[i_year], tile_name, year, 'SR_stable', src_geotrans, src_proj, output_path_root)

        output_temporal_feature_image(img_sr_change[i_year], tile_name, year, 'SR_change', src_geotrans, src_proj, output_path_root)

        output_temporal_feature_image(img_c1[i_year], tile_name, year, 'c1', src_geotrans, src_proj, output_path_root)

        output_temporal_feature_image(img_rmse[i_year], tile_name, year, 'RMSE', src_geotrans, src_proj, output_path_root)

        output_temporal_feature_image(img_a1[i_year], tile_name, year, 'a1', src_geotrans, src_proj, output_path_root)
        output_temporal_feature_image(img_b1[i_year], tile_name, year, 'b1', src_geotrans, src_proj, output_path_root)
        # output_temporal_feature_image(img_a2[i_year], tile_name, year, 'a2', src_geotrans, src_proj, output_path_root)
        # output_temporal_feature_image(img_b2[i_year], tile_name, year, 'b2', src_geotrans, src_proj, output_path_root)
        # output_temporal_feature_image(img_a3[i_year], tile_name, year, 'a3', src_geotrans, src_proj, output_path_root)
        # output_temporal_feature_image(img_b3[i_year], tile_name, year, 'b3', src_geotrans, src_proj, output_path_root)

        # output_temporal_feature_image(img_harmonic_coef[i_year], tile_name, year, 'Harmonic_coef', src_geotrans, src_proj, output_path_root)

        logging.info('Predictor variables extraction finished for tile {} year {}:'.format(tile_name, year))


#
# if __name__ == '__main__':
def main():

    tile_name = 'h003v000'

    path_cold = r'/shared/zhulab/Shi/ProjectCONUSDisturbanceAgent/Detection/'  # path of the COLD rec_cg file in Shi's shared folder
    # list_tile_name = os.listdir(path_cold)  # total tiles to cover the CONUS region
    # list_tile_name.sort()

    output_rootpath_project = r'/gpfs/sharedfs1/zhulab/Falu/CSM_project/'

    # log file
    path_log = join(output_rootpath_project, 'data', 'predictor_variable', tile_name)
    print(path_log)
    if not os.path.exists(path_log):
        os.makedirs(path_log, exist_ok=True)

    logging.basicConfig(filename=join(path_log, '{}.log'.format(tile_name)),
                        level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    for year in range(2022, 2023):
        list_year_extraction = np.arange(year, year + 1)
        pipeline_extract_temporal_feature(path_cold, tile_name, list_year_extraction,
                                          nan_fill=0, rows_running=5000,
                                          output_project_folder=output_rootpath_project,
                                          sr_stable_output_flag=False)  # stable (central) surface reflectance is not stored



