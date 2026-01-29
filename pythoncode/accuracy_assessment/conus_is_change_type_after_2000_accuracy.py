"""
    code to evaluate the accuracy of the IS change type map with the Good Practice strategy for the period of 2000 to 2020
"""

import os
from os.path import join
import sys
import pandas as pd
import numpy as np

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.accurace_assessment.utils_good_practice_accuracy_assessment import (generate_good_practice_matrix,
                                                                                    plot_df_confusion,
                                                                                    get_adjusted_area_and_margin_of_error)

from pythoncode.accurace_assessment.utils_conus_is_pct_change_type_assessment import (get_weight_for_whole_conus_is_change_type,
                                                                                      prepare_evaluation_data)


def prepare_interpretation_dataframe(sample_folder, isp_folder):
    """
        prepare the interpretation dataframe for evaluation

        :param sample_folder:
        :param isp_folder:
        :return:
    """

    path_sample = join(rootpath, 'results', 'accuracy_assessment', 'conus_is_change_type')
    output_filename_interpretation = join(path_sample, f'{sample_folder}_sample_interpretation.xlsx')
    df_sample_interpretation = pd.read_excel(output_filename_interpretation, sheet_name='interpretation_record')

    output_filename_map = join(path_sample, f'{sample_folder}_{isp_folder}_extract_output.xlsx')
    df_sample_map = pd.read_excel(output_filename_map, sheet_name='Sheet1')

    # insert the stratum column from the mapped dataframe to the interpretation dataframe
    df_interpretation_merge = df_sample_map.copy()
    df_interpretation_merge['interpretation_is_change_type'] = df_sample_interpretation['interpretation_is_change_type'].values

    return df_interpretation_merge


def get_map_reference_label_based_on_match_flag(df_interpretation_merge, match_flag, dict_is_change_type):
    """
    Determines the map reference label based on the given match flag ('exact_match' or 'buffer_match')
    prepare and return the final map and reference arrays for evaluation.

    For 'exact_match', the function directly prepares the evaluation data for the specified map column.
    For 'buffer_match', if the map type is within three years of the reference type, it is considered a match
    """

    if match_flag == 'exact_match':
        (array_map_final, array_reference_final) = prepare_evaluation_data(df_interpretation_merge, dict_is_change_type,
                                                                           map_column_name='stratum')

    elif match_flag == 'buffer_match':
        # buffer match means if the map type is within three years of the reference type, it is considered a match,
        # otherwise, the exact match is used

        (array_map_exact, array_reference_final) = prepare_evaluation_data(df_interpretation_merge, dict_is_change_type,
                                                                              map_column_name='stratum')

        (array_map_year_0_year_1, array_reference_final) = prepare_evaluation_data(df_interpretation_merge, dict_is_change_type,
                                                                                      map_column_name='map_type_year_0_to_year_1')

        (array_map_year_1_year_2, array_reference_final) = prepare_evaluation_data(df_interpretation_merge, dict_is_change_type,
                                                                                      map_column_name='map_type_year_2_to_year_3')

        array_map_final = np.zeros_like(array_map_exact, dtype=int)

        for p in range(0, len(array_map_final)):

            if ((array_map_exact[p] == array_reference_final[p]) or
                (array_map_year_0_year_1[p] == array_reference_final[p]) or
                (array_map_year_1_year_2[p] == array_reference_final[p])):

                array_map_final[p] = array_reference_final[p]
            else:
                array_map_final[p] = array_map_exact[p]

        # define the categories to avoid missing categories in the confusion matrix
        array_map_final = pd.Categorical(array_map_final, categories=np.arange(1, len(dict_is_change_type) + 1))

    return (array_map_final, array_reference_final)


# def main():
if __name__ == '__main__':

    sample_folder = 'v3_conus_2000_2020'
    match_flag = 'exact_match'  # exact_match or buffer_match (buffer match means agreement based on three-year window)
    print(f'match_flag: {match_flag}')
    print(f'processing {sample_folder}')

    data_flag = 'conus_isp'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    dict_is_change_type = {'1': 'stable natural',
                           '2': 'stable IS',
                           '3': 'IS expansion',
                           '4': 'IS intensification',
                           '5': 'IS decline',
                           '6': 'IS reversal',
                           '7': 'surface modification'}

    np.set_printoptions(suppress=True)

    df_interpretation_merge = prepare_interpretation_dataframe(sample_folder=sample_folder,
                                                               isp_folder=isp_folder)

    # get the count and weight
    (array_weight, array_count) = get_weight_for_whole_conus_is_change_type(array_target_year=np.arange(2000, 2021) ,
                                                                            data_flag=data_flag,
                                                                            isp_folder=isp_folder,
                                                                            rootpath_project_folder=None)

    # get the map and reference label based on the match flag
    (array_map_final, array_reference_final) = get_map_reference_label_based_on_match_flag(df_interpretation_merge,
                                                                                           match_flag,
                                                                                           dict_is_change_type)
    ##
    # (array_map_final, array_reference_final) = prepare_evaluation_data(df_interpretation_merge, dict_is_change_type)

    df_confusion_lc = pd.crosstab(array_map_final, array_reference_final, rownames=['Map'], colnames=['Reference'], dropna=False)
    overall_accuracy = np.trace(df_confusion_lc.values) / np.sum(df_confusion_lc.values)

    print(f'number of agreement pixels: {np.trace(df_confusion_lc.values)} / {df_confusion_lc.values.sum()}')
    print(f'count-based overall accuracy {overall_accuracy}')

    # area-based confusion matrix
    df_err_adjust = generate_good_practice_matrix(df_confusion_lc.values, array_weight, array_count)
    print('adjusted overall accuracy {}'.format(df_err_adjust.loc['PA', 'UA']))

    plot_df_confusion(df_confusion_lc.values,
                      stratum_des=dict_is_change_type,
                      title='IS change types', figsize=(12, 9))
    ##
    (array_mapped_area,
     array_adjusted_area,
     stand_error_area) = get_adjusted_area_and_margin_of_error(data=df_confusion_lc.values,
                                                               df_err_adjust=df_err_adjust,
                                                               array_count=array_count,
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
    #
    # if match_flag == 'exact_match':
    #
    #     output_filename = join(output_path, f'{sample_folder}_accuracy_report_output.xlsx')
    #
    #     with pd.ExcelWriter(output_filename) as writer:
    #         df_confusion_lc.to_excel(writer, sheet_name='sample_count_matrix')
    #         df_err_adjust.to_excel(writer, sheet_name='error_adjusted_matrix')
    #
    # elif match_flag == 'buffer_match':
    #     output_filename = join(output_path, f'{sample_folder}_accuracy_report_buffer_match_output.xlsx')
    #
    #     with pd.ExcelWriter(output_filename) as writer:
    #         df_confusion_lc.to_excel(writer, sheet_name='sample_count_matrix')
    #         df_err_adjust.to_excel(writer, sheet_name='error_adjusted_matrix')



