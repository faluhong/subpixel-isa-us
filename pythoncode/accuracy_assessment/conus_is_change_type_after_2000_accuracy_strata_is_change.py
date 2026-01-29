"""
    merge the stratum and report the accuracy for CONUS IS change type

    The aggregation/merge rule:
    no IS change: (1) stable natural, (2) stable IS, (3) surface modification
    IS change: (4) IS expansion, (5) IS intensification, (6) IS decline, (7) IS reversal
"""

import os
from os.path import join, exists
import sys
import pandas as pd
from osgeo import gdal, gdal_array, gdalconst
import numpy as np
from lulc_validation.lulc_val import StratVal
import warnings

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.accurace_assessment.utils_good_practice_accuracy_assessment import (plot_df_confusion,
                                                                                    get_adjusted_area_and_margin_of_error,
                                                                                    calculate_actual_pixel_after_starta_change,
                                                                                    get_area_corrected_df_after_strata_change)

from pythoncode.accurace_assessment.utils_conus_is_pct_change_type_assessment import (get_weight_for_whole_conus_is_change_type, )

from pythoncode.accurace_assessment.conus_is_change_type_after_2000_accuracy import (prepare_interpretation_dataframe,
                                                                                     get_map_reference_label_based_on_match_flag)


def merge_stratum_in_sample(array_sample, dict_is_change_type_merge):

    # adjust the strata in the sample
    array_map_merge = array_sample.copy()

    # IS no change, including (1) stable natural, (2) stable IS, (3) surface modification
    array_map_merge[np.isin(array_map_merge, [1, 2, 7])] = 1

    # IS change, including (4) IS expansion, (5) IS intensification, (6) IS decline, (7) IS reversal
    array_map_merge[np.isin(array_map_merge, [3, 4, 5, 6])] = 2

    # define the categories to avoid missing categories in the confusion matrix

    categories = np.arange(1, len(dict_is_change_type_merge) + 1)

    array_map_merge = pd.Categorical(array_map_merge, categories=categories)

    return array_map_merge


def report_strata_change_accuracy(array_ori_strata,
                                  array_new_map,
                                  array_new_reference,
                                  list_count_original_stata,
                                  dict_is_change_type_merge,
                                  dict_is_change_type):
    """
        Report the accuracy of the new IS change map after changing the strata

        :param array_ori_strata: array of original strata label
        :param array_new_map: array of map label for new strata
        :param array_new_reference: array of reference label for new strata
        :param list_count_original_stata: pixel count of original strata
        :param dict_is_change_type_merge:
        :param dict_is_change_type:
        :return:
    """

    df_confusion_lc = pd.crosstab(array_new_map, array_new_reference,
                                  rownames=['Map'], colnames=['Reference'], dropna=False)
    overall_accuracy = np.trace(df_confusion_lc.values) / np.sum(df_confusion_lc.values)

    print(f'number of agreement pixels: {np.trace(df_confusion_lc.values)} / {df_confusion_lc.values.sum()}')
    print(f'count-based overall accuracy {overall_accuracy}')

    # example in "strata change" paper
    df_sample = pd.DataFrame({'strata': array_ori_strata,
                              'map_class': array_new_map,
                              'ref_class': array_new_reference,
                              })

    # list_count_original_stata = array_count.copy()

    strata_change_val = StratVal(
        strata_list=[int(p) for p in dict_is_change_type.keys()],  # List of labels for strata.
        class_list=[int(p) for p in dict_is_change_type_merge.keys()],  # List of labels for LULC map classes.
        n_strata=list_count_original_stata,  # array of the total number of pixels in each stratum.
        samples_df=df_sample,  # pandas DataFrame of reference data
        strata_col="strata",  # Column label for strata in `samples_df`
        ref_class="ref_class",  # Column label for reference classes in `samples_df`
        map_class="map_class"  # Column label for map classes in `samples_df`
    )

    # overall_accuracy = strata_change_val.accuracy()
    # users_accuracy = strata_change_val.users_accuracy()
    # producers_accuracy = strata_change_val.producers_accuracy()
    # overall_accuracy_se = strata_change_val.accuracy_se()
    # users_accuracy_se = strata_change_val.users_accuracy_se()
    # producers_accuracy_se = strata_change_val.producers_accuracy_se()

    print(f"accuracy: {strata_change_val.accuracy()}")
    print(f"user's accuracy: {strata_change_val.users_accuracy()}")
    print(f"producer's accuracy: {strata_change_val.producers_accuracy()}")
    print(f"accuracy se: {strata_change_val.accuracy_se()}")
    print(f"user's accuracy se: {strata_change_val.users_accuracy_se()}")
    print(f"producers's accuracy se: {strata_change_val.producers_accuracy_se()}")

    array_count_actual_pixel_after_strata_change = calculate_actual_pixel_after_starta_change(array_map_original=df_sample['strata'].values,
                                                                                              array_map_updated=df_sample['map_class'].values,
                                                                                              array_reference=df_sample['ref_class'].values,
                                                                                              array_count_original_stratum=list_count_original_stata,
                                                                                              stratum_count_original=len(list_count_original_stata),
                                                                                              stratum_count_new=len(list_count_original_stata))

    df_err_adjust_strata_change = get_area_corrected_df_after_strata_change(array_count_actual_pixel_after_strata_change,
                                                                            strata_change_val,
                                                                            confidence_interval=1.96,
                                                                            new_stratum_count=len(dict_is_change_type_merge))

    return (df_sample, df_confusion_lc, df_err_adjust_strata_change)


# def main():
if __name__ == '__main__':

    sample_folder = 'v3_conus_2000_2020'
    match_flag = 'buffer_match'  # exact_match or buffer_match (buffer match means agreement based on three-year window)
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

    #  prepare the interpretation dataframe for analysis
    df_interpretation_merge = prepare_interpretation_dataframe(sample_folder=sample_folder,
                                                               isp_folder=isp_folder)

    # get the count and weight
    (array_weight, array_count) = get_weight_for_whole_conus_is_change_type(array_target_year=np.arange(2000, 2021) ,
                                                                            data_flag=data_flag,
                                                                            isp_folder=isp_folder,
                                                                            rootpath_project_folder=None)

    (array_map_ori_strata,
     array_reference_ori_strata) = get_map_reference_label_based_on_match_flag(df_interpretation_merge,
                                                                               match_flag=match_flag,
                                                                               dict_is_change_type=dict_is_change_type)

    ##
    dict_is_change_type_merge = {'1': 'no IS change',
                                 '2': 'IS change',}

    array_map_merge = merge_stratum_in_sample(array_sample=array_map_ori_strata,
                                              dict_is_change_type_merge=dict_is_change_type_merge)
    array_reference_merge = merge_stratum_in_sample(array_sample=array_reference_ori_strata,
                                                    dict_is_change_type_merge=dict_is_change_type_merge)

    # adjust the weight and count
    array_weight_merge = np.zeros([2], dtype=float)
    array_weight_merge[0] = np.sum(array_weight[[0, 1, 6]])
    array_weight_merge[1] = np.sum(array_weight[[2, 3, 4, 5]])

    array_count_merge = np.zeros([2], dtype=float)
    array_count_merge[0] = np.sum(array_count[[0, 1, 6]])
    array_count_merge[1] = np.sum(array_count[[2, 3, 4, 5]])

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

    # print area, noted that this is pixel count based area
    # np.set_printoptions(suppress=True)
    #
    # for p in array_mapped_area:
    #     print(f'mapped area: {p:.1f} km2')
    # for p in array_adjusted_area:
    #     print(f'adjusted area: {p:.1f} km2')

    ##
    # output_path = join(rootpath, 'results', 'accuracy_assessment', 'conus_is_change_type')

    # if match_flag == 'exact_match':
    #     output_filename = join(output_path, f'{sample_folder}_accuracy_report_strata_merge_output.xlsx')
    #
    #     with pd.ExcelWriter(output_filename) as writer:
    #         df_confusion_lc.to_excel(writer, sheet_name='sample_count_matrix')
    #         df_err_adjust_strata_change.to_excel(writer, sheet_name='error_adjusted_matrix')
    #
    # elif match_flag == 'buffer_match':
    #     output_filename = join(output_path, f'{sample_folder}_accuracy_report_strata_merge_buffer_match_output.xlsx')
    #
    #     with pd.ExcelWriter(output_filename) as writer:
    #         df_confusion_lc.to_excel(writer, sheet_name='sample_count_matrix')
    #         df_err_adjust_strata_change.to_excel(writer, sheet_name='error_adjusted_matrix')



