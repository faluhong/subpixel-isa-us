"""
    merge the stratum and report the accuracy for CONUS IS change type

    The strata after merging include:
    (1) IS expansion
    (2) IS intensification
    (3) Other: IS decline, IS reversal, stable natural, stable IS, surface modification
"""

import os
from os.path import join, exists
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array, gdalconst
import matplotlib
import numpy as np
from lulc_validation.lulc_val import StratVal
import warnings

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from sample_based_analysis.utils_good_practice_accuracy_assessment  import (generate_good_practice_matrix,
                                                                            plot_df_confusion,
                                                                            get_adjusted_area_and_margin_of_error,
                                                                            calculate_actual_pixel_after_starta_change,
                                                                            get_area_corrected_df_after_strata_change)

from sample_based_analysis.utils_conus_is_pct_change_type_assessment import (get_weight_for_whole_conus_is_change_type,
                                                                             prepare_evaluation_data)

from sample_based_analysis.conus_is_change_type_after_2000_accuracy import (get_map_reference_label_based_on_match_flag)


from sample_based_analysis.conus_is_change_type_after_2000_accuracy_merge import (report_strata_change_accuracy)


def merge_stratum_in_sample_focus_on_is_expansion_intensification(array_sample, dict_is_change_type_merge):
    """
        Get the new strata after merging the stratum in the sample

        :param array_sample:
        :param dict_is_change_type_merge:
        :return:
    """

    # adjust the strata in the sample
    array_map_merge = array_sample.copy()
    array_map_merge = array_map_merge.astype(int)

    # Other: stable natural (1), stable IS (2), IS decline (5), IS reversal (6), surface modification (7)
    array_map_merge[np.isin(array_map_merge, [1, 2, 5, 6, 7])] = 11

    # Change the IS expansion from 3 to 2
    array_map_merge[np.isin(array_map_merge, [3])] = 12

    # Change the IS intensification from 4 to 3
    array_map_merge[np.isin(array_map_merge, [4])] = 13

    # renumber the strata. 1: IS expansion, 2: IS intensification, 3: Other
    array_map_merge[array_map_merge == 11] = 3
    array_map_merge[array_map_merge == 12] = 1
    array_map_merge[array_map_merge == 13] = 2

    # define the categories to avoid missing categories in the confusion matrix
    categories = np.arange(1, len(dict_is_change_type_merge) + 1)

    array_map_merge = pd.Categorical(array_map_merge, categories=categories)

    return array_map_merge


# def main():
if __name__ == '__main__':

    sample_folder = 'v3_conus_2000_2020'
    match_flag = 'exact_match'  # exact_match or buffer_match (buffer match means agreement based on three-year window)
    print(f'match_flag: {match_flag}')
    print(f'processing {sample_folder}')

    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    warnings.filterwarnings('ignore')

    dict_is_change_type = {'1': 'stable natural',
                           '2': 'stable IS',
                           '3': 'IS expansion',
                           '4': 'IS intensification',
                           '5': 'IS decline',
                           '6': 'IS reversal',
                           '7': 'surface modification'}

    np.set_printoptions(suppress=True)

    output_path = join(rootpath, 'results', 'accuracy_assessment', 'conus_is_change_type', sample_folder)

    # prepare the interpretation dataframe for analysis
    output_filename_map = join(output_path, f'{sample_folder}_{isp_folder}_extract_output.xlsx')
    df_sample_map = pd.read_excel(output_filename_map, sheet_name='Sheet1')

    output_filename_interpretation = join(output_path, f'{sample_folder}_sample_interpretation.xlsx')
    df_sample_interpretation = pd.read_excel(output_filename_interpretation, sheet_name='interpretation_record')

    # insert the stratum column from the mapped dataframe to the interpretation dataframe
    df_interpretation_merge = df_sample_map.copy()
    df_interpretation_merge['interpretation_is_change_type'] = df_sample_interpretation['interpretation_is_change_type'].values

    # get the count and weight
    (array_weight, array_count) = get_weight_for_whole_conus_is_change_type(array_target_year=np.arange(2000, 2021) ,
                                                                            data_flag=data_flag,
                                                                            isp_folder=isp_folder,
                                                                            rootpath_project_folder=None)

    (array_map_ori_strata, array_reference_ori_strata) = get_map_reference_label_based_on_match_flag(df_interpretation_merge,
                                                                                                     match_flag=match_flag,
                                                                                                     dict_is_change_type=dict_is_change_type)

    ##

    # 1: IS expansion, 2: IS intensification, 3: Other

    dict_is_change_type_merge = {'1': 'IS expansion',
                                 '2': 'IS intensification',
                                 '3': 'Other',}

    array_map_merge = merge_stratum_in_sample_focus_on_is_expansion_intensification(array_sample=array_map_ori_strata,
                                                                                    dict_is_change_type_merge=dict_is_change_type_merge)
    array_reference_merge = merge_stratum_in_sample_focus_on_is_expansion_intensification(array_sample=array_reference_ori_strata,
                                                                                          dict_is_change_type_merge=dict_is_change_type_merge)

    # adjust the weight and count
    array_weight_merge = np.zeros([len(dict_is_change_type_merge)], dtype=float)
    array_weight_merge[0] = array_weight[2]
    array_weight_merge[1] = array_weight[3]
    array_weight_merge[2] = np.sum(array_weight[[0, 1, 4, 5, 6]])

    array_count_merge = np.zeros([len(dict_is_change_type_merge)], dtype=float)
    array_count_merge[0] = array_count[2]
    array_count_merge[1] = array_count[3]
    array_count_merge[2] = np.sum(array_count[[0, 1, 4, 5, 6]])

    ##
    (df_sample,
     df_confusion_lc,
     df_err_adjust_strata_change) = report_strata_change_accuracy(array_ori_strata=array_map_ori_strata,
                                                                  array_new_map=array_map_merge,
                                                                  array_new_reference=array_reference_merge,
                                                                  list_count_original_stata=array_count,
                                                                  dict_is_change_type_merge=dict_is_change_type_merge,
                                                                  dict_is_change_type=dict_is_change_type)

    plot_df_confusion(df_confusion_lc.values,
                      stratum_des=dict_is_change_type_merge,
                      title='IS change types', figsize=(12, 9))

    ##
    df_err_adjust_strata_change['weight'] = np.nan
    df_err_adjust_strata_change.iloc[0: len(array_weight_merge), -1] = array_weight_merge

    (array_mapped_area,
     array_adjusted_area,
     stand_error_area) = get_adjusted_area_and_margin_of_error(data=df_confusion_lc.values,
                                                               df_err_adjust=df_err_adjust_strata_change,
                                                               array_count=array_count_merge,
                                                               confidence_interval=1.96)

    ##
    np.set_printoptions(suppress=True)

    for p in array_mapped_area:
        print(f'mapped area: {p:.1f} km2')
    for p in array_adjusted_area:
        print(f'adjusted area: {p:.1f} km2')

    ##
    df_err_adjust_strata_change = df_err_adjust_strata_change * 100  # convert to percentage

    if match_flag == 'exact_match':
        output_filename = join(output_path, f'{sample_folder}_accuracy_report_strata_merge_is_expansion_intensification_output.xlsx')

        with pd.ExcelWriter(output_filename) as writer:
            df_confusion_lc.to_excel(writer, sheet_name='sample_count_matrix')
            df_err_adjust_strata_change.to_excel(writer, sheet_name='error_adjusted_matrix')

    elif match_flag == 'buffer_match':
        output_filename = join(output_path, f'{sample_folder}_accuracy_report_strata_merge_is_expansion_intensification_buffer_match_output.xlsx')

        with pd.ExcelWriter(output_filename) as writer:
            df_confusion_lc.to_excel(writer, sheet_name='sample_count_matrix')
            df_err_adjust_strata_change.to_excel(writer, sheet_name='error_adjusted_matrix')







