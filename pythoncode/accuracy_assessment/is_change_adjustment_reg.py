"""
    adjust the IS change area using all-sample regression results
"""

import numpy as np
import os
from os.path import join, exists
import sys

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.accurace_assessment.utils_conus_is_area_uncertainty_est import (get_conus_is_change_summary_data)

from pythoncode.conus_isa_financial_crisis_resilience.utils_print_red_rec import (print_is_reduction_recovery,
                                                                                  print_adjust_is_area_change_stats)

from pythoncode.accurace_assessment.is_area_adjustment_reg_all_sample import (get_weight_regression_results,
                                                                              cal_isp_reg_confidence_interval)

from pythoncode.conus_isa_analysis.utils_plot_is_area import (manuscript_is_area_plot)



def extract_array_of_is_changes(img_sum_isp_change_stats_diag_zero_conus_1988_2020):
    """
        extract the array of IS changes from 1988 to 2020

        :param img_sum_isp_change_stats_diag_zero_conus_1988_2020: the array recording the IS change stats from 1988 to 2020

        :return:
    """

    # expansion
    array_expansion_intensity = np.concatenate([
        np.full(int(img_sum_isp_change_stats_diag_zero_conus_1988_2020[0, p]), p)
        for p in range(1, 101)
    ])

    # reversal
    array_reversal_intensity = -np.concatenate([
        np.full(int(img_sum_isp_change_stats_diag_zero_conus_1988_2020[p, 0]), p)
        for p in range(1, 101)
    ])

    # intensification
    array_intensification_start = np.concatenate([
        np.full(int(img_sum_isp_change_stats_diag_zero_conus_1988_2020[p, q]), p)
        for p in range(1, 101) for q in range(p + 1, 101)
    ])

    array_intensification_end = np.concatenate([
        np.full(int(img_sum_isp_change_stats_diag_zero_conus_1988_2020[p, q]), q)
        for p in range(1, 101) for q in range(p + 1, 101)
    ])

    array_intensification_intensity = array_intensification_end - array_intensification_start

    # decline
    array_decline_start = np.concatenate([
        np.full(int(img_sum_isp_change_stats_diag_zero_conus_1988_2020[q, p]), q)
        for p in range(1, 101) for q in range(p + 1, 101)
    ])

    array_decline_end = np.concatenate([
        np.full(int(img_sum_isp_change_stats_diag_zero_conus_1988_2020[q, p]), p)
        for p in range(1, 101) for q in range(p + 1, 101)
    ])

    array_decline_intensity = array_decline_end - array_decline_start

    return (array_expansion_intensity, array_reversal_intensity,
            array_intensification_start, array_intensification_end, array_intensification_intensity,
            array_decline_start, array_decline_end, array_decline_intensity
            )


def generate_adjusted_conus_is_change_dataframe(df_is_change_sum_conus_plot,
                                                weight_reg_summary,
                                                img_sum_isp_change_stats_diag_zero_conus_plot):
    """
        generate the adjusted CONUS IS change dataframe based on regression results

        :param df_is_change_sum_conus_plot:
        :param weight_reg_summary:
        :param img_sum_isp_change_stats_diag_zero_conus_plot:
        :return:
    """

    # adjust the IS area and change area with uncertainty interval

    df_is_change_sum_conus_plot_adjust = df_is_change_sum_conus_plot.copy()
    df_is_change_sum_conus_plot_adjust = df_is_change_sum_conus_plot_adjust.reset_index(drop=True)

    conus_total_area = df_is_change_sum_conus_plot_adjust['total_area'].values[0]

    array_is_pct_year_1 = df_is_change_sum_conus_plot_adjust['is_area_year_1'].values / conus_total_area * 100.0
    array_is_pct_year_2 = df_is_change_sum_conus_plot_adjust['is_area_year_2'].values / conus_total_area * 100.0

    (array_is_pct_year_1,
     array_is_pct_year_1_adjust,
     ci_upper_is_pct_year_1,
     ci_lower_is_pct_year_1,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                x_input=array_is_pct_year_1)

    (array_is_pct_year_2,
     array_is_pct_year_2_adjust,
     ci_upper_is_pct_year_2,
     ci_lower_is_pct_year_2,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                x_input=array_is_pct_year_2)

    df_is_change_sum_conus_plot_adjust['is_area_year_1_adjust'] = array_is_pct_year_1_adjust / 100.0 * conus_total_area
    df_is_change_sum_conus_plot_adjust['is_area_year_2_adjust'] = array_is_pct_year_2_adjust / 100.0 * conus_total_area

    df_is_change_sum_conus_plot_adjust['is_area_year_1_ci_lower'] = ci_lower_is_pct_year_1 / 100.0 * conus_total_area
    df_is_change_sum_conus_plot_adjust['is_area_year_1_ci_upper'] = ci_upper_is_pct_year_1 / 100.0 * conus_total_area
    df_is_change_sum_conus_plot_adjust['is_area_year_2_ci_lower'] = ci_lower_is_pct_year_2 / 100.0 * conus_total_area
    df_is_change_sum_conus_plot_adjust['is_area_year_2_ci_upper'] = ci_upper_is_pct_year_2 / 100.0 * conus_total_area

    # insert columns for adjusted IS change components
    df_is_change_sum_conus_plot_adjust['area_is_expansion_adjust'] = np.nan
    df_is_change_sum_conus_plot_adjust['area_is_expansion_ci_upper'] = np.nan
    df_is_change_sum_conus_plot_adjust['area_is_expansion_ci_lower'] = np.nan

    df_is_change_sum_conus_plot_adjust['area_is_intensification_adjust'] = np.nan
    df_is_change_sum_conus_plot_adjust['area_is_intensification_ci_upper'] = np.nan
    df_is_change_sum_conus_plot_adjust['area_is_intensification_ci_lower'] = np.nan

    df_is_change_sum_conus_plot_adjust['area_is_decline_adjust'] = np.nan
    df_is_change_sum_conus_plot_adjust['area_is_decline_ci_upper'] = np.nan
    df_is_change_sum_conus_plot_adjust['area_is_decline_ci_lower'] = np.nan

    df_is_change_sum_conus_plot_adjust['area_is_reversal_adjust'] = np.nan
    df_is_change_sum_conus_plot_adjust['area_is_reversal_ci_upper'] = np.nan
    df_is_change_sum_conus_plot_adjust['area_is_reversal_ci_lower'] = np.nan

    df_is_change_sum_conus_plot_adjust['total_is_area_change_adjust'] = np.nan
    df_is_change_sum_conus_plot_adjust['total_is_area_change_ci_upper'] = np.nan
    df_is_change_sum_conus_plot_adjust['total_is_area_change_ci_lower'] = np.nan

    # adjust the year-to-year change area
    for i_year in range(0, len(df_is_change_sum_conus_plot_adjust)):
        img_sum_isp_change_stats_diag_zero_conus_single_year = img_sum_isp_change_stats_diag_zero_conus_plot[i_year, :, :]

        img_sum_isp_change_stats_diag_zero_conus_single_year[np.isnan(img_sum_isp_change_stats_diag_zero_conus_single_year)] = 0

        count_expansion = df_is_change_sum_conus_plot_adjust['count_is_expansion'].values[i_year]
        count_reversal = df_is_change_sum_conus_plot_adjust['count_is_reversal'].values[i_year]
        count_intensification = df_is_change_sum_conus_plot_adjust['count_is_intensification'].values[i_year]
        count_decline = df_is_change_sum_conus_plot_adjust['count_is_decline'].values[i_year]

        (array_expansion_intensity,
         array_reversal_intensity,  # negative values
         array_intensification_start,
         array_intensification_end,
         array_intensification_intensity,
         array_decline_start,
         array_decline_end,
         array_decline_intensity) = extract_array_of_is_changes(img_sum_isp_change_stats_diag_zero_conus_single_year)

        (expansion_start_ori,
         expansion_start_adjust,
         ci_upper_expansion_start,
         ci_lower_expansion_start,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                      x_input=0)

        (expansion_end_ori,
         expansion_end_adjust,
         ci_upper_expansion_end,
         ci_lower_expansion_end,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                    x_input=np.nanmean(array_expansion_intensity))

        (intensification_start_ori,
         intensification_start_adjust,
         ci_upper_intensification_start,
         ci_lower_intensification_start,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                            x_input=np.nanmean(array_intensification_start))

        (intensification_end_ori,
         intensification_end_adjust,
         ci_upper_intensification_end,
         ci_lower_intensification_end,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                          x_input=np.nanmean(array_intensification_end))

        (decline_start_ori,
         decline_start_adjust,
         ci_upper_decline_start,
         ci_lower_decline_start,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                    x_input=np.nanmean(array_decline_start))

        (decline_end_ori,
         decline_end_adjust,
         ci_upper_decline_end,
         ci_lower_decline_end,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                  x_input=np.nanmean(array_decline_end))

        (reversal_start_ori,
         reversal_start_adjust,
         ci_upper_reversal_start,
         ci_lower_reversal_start,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                     x_input=np.nanmean(np.abs(array_reversal_intensity)))

        (reversal_end_ori,
         reversal_end_adjust,
         ci_upper_reversal_end,
         ci_lower_reversal_end,) = cal_isp_reg_confidence_interval(weight_reg_summary,
                                                                   x_input=0)

        area_expansion_adjust = (expansion_end_adjust - expansion_start_adjust) * count_expansion * 900 / 100
        area_expansion_ci_upper = (ci_upper_expansion_end - ci_lower_expansion_start) * count_expansion * 900 / 100
        area_expansion_ci_lower = (ci_lower_expansion_end - ci_upper_expansion_start) * count_expansion * 900 / 100

        area_intensification_adjust = (intensification_end_adjust - intensification_start_adjust) * count_intensification * 900 / 100
        area_intensification_ci_upper = (ci_upper_intensification_end - ci_lower_intensification_start) * count_intensification * 900 / 100
        area_intensification_ci_lower = (ci_lower_intensification_end - ci_upper_intensification_start) * count_intensification * 900 / 100

        area_reversal_adjust = (reversal_end_adjust - reversal_start_adjust) * count_reversal * 900 / 100
        area_reversal_ci_upper = (ci_upper_reversal_end - ci_lower_reversal_start) * count_reversal * 900 / 100
        area_reversal_ci_lower = (ci_lower_reversal_end - ci_upper_reversal_start) * count_reversal * 900 / 100

        # convert to absolute values
        area_reversal_adjust = np.abs(area_reversal_adjust)
        area_reversal_ci_upper = np.abs(area_reversal_ci_upper)
        area_reversal_ci_lower = np.abs(area_reversal_ci_lower)

        area_decline_adjust = (decline_end_adjust - decline_start_adjust) * count_decline * 900 / 100
        area_decline_ci_upper = (ci_upper_decline_end - ci_lower_decline_start) * count_decline * 900 / 100
        area_decline_ci_lower = (ci_lower_decline_end - ci_upper_decline_start) * count_decline * 900 / 100

        # convert to absolute values
        area_decline_adjust = np.abs(area_decline_adjust)
        area_decline_ci_upper = np.abs(area_decline_ci_upper)
        area_decline_ci_lower = np.abs(area_decline_ci_lower)

        # convert to 0 if count is 0 to avoid nan results
        if count_expansion == 0:
            area_expansion_adjust = 0.0
            area_expansion_ci_upper = 0.0
            area_expansion_ci_lower = 0.0
        if count_intensification == 0:
            area_intensification_adjust = 0.0
            area_intensification_ci_upper = 0.0
            area_intensification_ci_lower = 0.0
        if count_reversal == 0:
            area_reversal_adjust = 0.0
            area_reversal_ci_upper = 0.0
            area_reversal_ci_lower = 0.0
        if count_decline == 0:
            area_decline_adjust = 0.0
            area_decline_ci_upper = 0.0
            area_decline_ci_lower = 0.0

        area_total_change_adjust = (array_is_pct_year_2_adjust[i_year] - array_is_pct_year_1_adjust[i_year]) / 100.0 * conus_total_area
        assert np.abs(area_total_change_adjust - (area_expansion_adjust + area_intensification_adjust - area_reversal_adjust - area_decline_adjust)) < 1e-3, \
            f'area change not match at year index {i_year}'

        area_total_change_ci_upper = (area_expansion_ci_upper + area_intensification_ci_upper - area_reversal_ci_upper - area_decline_ci_upper)
        area_total_change_ci_lower = (area_expansion_ci_lower + area_intensification_ci_lower - area_reversal_ci_lower - area_decline_ci_lower)

        # different from the above one, this one has narrow 95% CI
        # area_total_change_ci_upper = (ci_upper_is_pct_year_2[i_year] - ci_upper_is_pct_year_1[i_year]) / 100.0 * conus_total_area
        # area_total_change_ci_lower = (ci_lower_is_pct_year_2[i_year] - ci_lower_is_pct_year_1[i_year]) / 100.0 * conus_total_area

        # print(area_total_change)
        # print((area_expansion_adjust + area_intensification_adjust + area_reversal_adjust + area_decline_adjust))
        # print('---')

        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_expansion_adjust'] = area_expansion_adjust
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_expansion_ci_upper'] = area_expansion_ci_upper
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_expansion_ci_lower'] = area_expansion_ci_lower

        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_intensification_adjust'] = area_intensification_adjust
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_intensification_ci_upper'] = area_intensification_ci_upper
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_intensification_ci_lower'] = area_intensification_ci_lower

        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_decline_adjust'] = area_decline_adjust
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_decline_ci_upper'] = area_decline_ci_upper
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_decline_ci_lower'] = area_decline_ci_lower

        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_reversal_adjust'] = area_reversal_adjust
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_reversal_ci_upper'] = area_reversal_ci_upper
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'area_is_reversal_ci_lower'] = area_reversal_ci_lower

        df_is_change_sum_conus_plot_adjust.loc[i_year, 'total_is_area_change_adjust'] = area_total_change_adjust
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'total_is_area_change_ci_upper'] = area_total_change_ci_upper
        df_is_change_sum_conus_plot_adjust.loc[i_year, 'total_is_area_change_ci_lower'] = area_total_change_ci_lower

    return df_is_change_sum_conus_plot_adjust


# def main():
if __name__ =='__main__':

    data_flag = 'conus_isp'   # 'conus_isp' or 'annual_nlcd'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'
    title = 'CONUS ISP'

    # array_year_plot = np.arange(1980, 2025, 1)
    array_year_plot = np.arange(1988, 2021, 1)

    (x, y, results) = get_weight_regression_results(data_flag=data_flag,
                                                    isp_folder=isp_folder)
    output_folder = join(rootpath, 'results', 'isp_change_stats', 'conus_summary', data_flag, isp_folder)

    print(data_flag, isp_folder, title)

    (df_is_change_sum_conus_plot,
     img_sum_isp_change_stats_conus_plot,
     img_sum_isp_change_stats_diag_zero_conus_plot,) = get_conus_is_change_summary_data(output_folder=output_folder,
                                                                                        array_year_plot=array_year_plot)

    df_is_change_sum_conus_plot_adjust = generate_adjusted_conus_is_change_dataframe(df_is_change_sum_conus_plot,
                                                                                     results,
                                                                                     img_sum_isp_change_stats_diag_zero_conus_plot)

    ##
    print('Mapped CONUS IS area change stats summary:')
    print_is_reduction_recovery(df_is_change_sum_conus_plot)
    print()

    print('Adjusted CONUS IS area change stats summary:')
    print_adjust_is_area_change_stats(df_is_change_sum_conus_plot_adjust=df_is_change_sum_conus_plot_adjust)

    ##
    title_1 = f'{title}: IS area change'
    title_2 = f'{title}: year-to-year IS area change'

    figsize = (25, 7)
    xlim = None
    ylim_1 = None
    ylim_2 = None

    if data_flag == 'conus_isp':
        output_filename = r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend\CONUS_IS_area_change.jpg'
    elif data_flag == 'annual_nlcd':
        output_filename = r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend\Annual_NLCD_IS_area_change.jpg'
    sns_style = 'white'
    legend_flag = False
    plot_flag = 'area'
    flag_highlight_2008 = False

    manuscript_is_area_plot(df_is_change_sum_conus_plot,
                            title_1, title_2,
                            figsize=figsize,
                            xlim=xlim, ylim_1=ylim_1, ylim_2=ylim_2,
                            output_flag=False,
                            output_filename=output_filename,
                            sns_style=sns_style,
                            legend_flag=legend_flag,
                            plot_flag=plot_flag,
                            flag_highlight_2008=flag_highlight_2008,
                            flag_focus_on_growth_types=True)

    # adjust the IS area and plot with uncertainty interval
    output_filename_adjust = r'C:\Users\64937\OneDrive\CSM_project\manuscript\figure\IS_area_trend\CONUS_IS_area_change_adjust.jpg'

    manuscript_is_area_plot(df_is_change_sum_conus_plot_adjust,
                            title_1,
                            title_2,
                            figsize=figsize,
                            xlim=xlim, ylim_1=ylim_1, ylim_2=ylim_2,
                            output_flag=False,
                            output_filename=output_filename_adjust,
                            sns_style=sns_style,
                            legend_flag=legend_flag,
                            plot_flag=plot_flag,
                            flag_highlight_2008=flag_highlight_2008,
                            flag_adjust_with_ci=True,
                            fill_alpha=0.3,
                            flag_focus_on_growth_types=True,
                            )

    ##








