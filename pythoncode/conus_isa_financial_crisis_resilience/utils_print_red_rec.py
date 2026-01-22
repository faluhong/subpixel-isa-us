"""
    utility functions to print the IS area change reduction and recovery statistics
"""

import numpy as np
import os
from os.path import join
import sys

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def print_is_reduction_recovery(df_is_change_sum_conus_plot):
    """
        print the IS increase reduction and recovery after the 2008 financial crisis

        2005-2008: before the financial crisis
        2008-2011: after the financial crisis
        2017-2020: recovery period

        print the total (net) IS increase reduction and recovery
        also print the IS expansion, intensification, decline, and reversal area changes reduction and recovery

        :param df_is_change_sum_conus_plot:
        :return:
    """

    # average IS growth rate from 1988 to 2008
    average_total_is_area_increase_2005_2008 = df_is_change_sum_conus_plot['total_is_area_change'][
                                                   (df_is_change_sum_conus_plot['year_1'] >= 2005) & (df_is_change_sum_conus_plot['year_1'] < 2008)].mean() / 1000 / 1000
    average_total_is_area_increase_2008_2011 = df_is_change_sum_conus_plot['total_is_area_change'][
                                                   (df_is_change_sum_conus_plot['year_1'] >= 2008) & (df_is_change_sum_conus_plot['year_1'] < 2011)].mean() / 1000 / 1000

    print(f'Average total IS area increase from 2005 to 2008: {average_total_is_area_increase_2005_2008:.4f} km^2/year')
    print(f'Average total IS area increase from 2008 to 2011: {average_total_is_area_increase_2008_2011:.4f} km^2/year')

    is_rate_reduction = (average_total_is_area_increase_2008_2011 - average_total_is_area_increase_2005_2008) / average_total_is_area_increase_2005_2008

    print(f'Average total IS area increase reduction: {is_rate_reduction * 100:.4f} %')

    average_area_is_area_increase_2017_2020 = df_is_change_sum_conus_plot['total_is_area_change'][
                                                   (df_is_change_sum_conus_plot['year_1'] >= 2017) & (df_is_change_sum_conus_plot['year_1'] < 2020)].mean() / 1000 / 1000
    print(f'Average total IS area increase from 2017 to 2020: {average_area_is_area_increase_2017_2020:.4f} km^2/year')

    is_rate_recovery = average_area_is_area_increase_2017_2020 / average_total_is_area_increase_2005_2008
    print(f'Average total IS area increase recovery: {is_rate_recovery * 100:.4f} %')


def print_adjust_is_area_change_stats(df_is_change_sum_conus_plot_adjust):
    """
        Print the adjusted IS area and changes with 95% confidence interval, including:

        (1) IS area and pct in 1988 with 95% confidence interval
        (2) IS area and pct in 2020 with 95% confidence interval
        (3) IS area change and pct change from 1988 to 2020 with 95% confidence interval

        (4) IS area change from 2005 to 2008 with 95% confidence interval
        (5) IS area change from 2008 to 2011 with 95% confidence interval
        (6) IS area change from 2017 to 2020 with 95% confidence interval

        (7) Reduction percentage with 95% confidence interval
        (8) Recovery percentage with 95% confidence interval

        :param df_is_change_sum_conus_plot_adjust:
        :return:
    """

    total_area = df_is_change_sum_conus_plot_adjust['total_area'].values[0] / 1e6  # convert to km2

    is_area_adjusted_1988_2020 = np.concatenate([df_is_change_sum_conus_plot_adjust['is_area_year_1_adjust'].values,
                                                 df_is_change_sum_conus_plot_adjust['is_area_year_2_adjust'].values[-1:]])
    is_area_adjusted_1988_2020 = is_area_adjusted_1988_2020 / 1e6  # convert to km2
    is_pct = is_area_adjusted_1988_2020 / total_area * 100

    is_area_adjusted_ci_upper = np.concatenate([df_is_change_sum_conus_plot_adjust['is_area_year_1_ci_upper'].values,
                                                df_is_change_sum_conus_plot_adjust['is_area_year_2_ci_upper'].values[-1:]])
    is_area_adjusted_ci_upper = is_area_adjusted_ci_upper / 1e6  # convert to km2
    is_pct_upper_ci = is_area_adjusted_ci_upper / total_area * 100

    is_area_adjusted_ci_lower = np.concatenate([df_is_change_sum_conus_plot_adjust['is_area_year_1_ci_lower'].values,
                                                df_is_change_sum_conus_plot_adjust['is_area_year_2_ci_lower'].values[-1:]])
    is_area_adjusted_ci_lower = is_area_adjusted_ci_lower / 1e6
    is_pct_lower_ci = is_area_adjusted_ci_lower / total_area * 100

    assert np.allclose(is_area_adjusted_ci_upper - is_area_adjusted_1988_2020, is_area_adjusted_1988_2020 - is_area_adjusted_ci_lower,
                       rtol=0.1, atol=1), 'CI calculation error'

    print(f'Adjusted IS area in 1988 with 95% CI')
    print(f'1988: {is_area_adjusted_1988_2020[0]:.0f}±{(is_area_adjusted_ci_upper[0] - is_area_adjusted_ci_lower[0]) / 2:.2f} km2 ')
    print(f'1988: {is_pct[0]:.2f} ±{(is_pct_upper_ci[0] - is_pct_lower_ci[0]) / 2:.4f}% of total area')

    print(f'Adjusted IS area in 2020 with 95% CI')
    print(f'2020: {is_area_adjusted_1988_2020[-1]:.0f}±{(is_area_adjusted_ci_upper[-1] - is_area_adjusted_ci_lower[-1]) / 2:.2f} km2 ')
    print(f'2020: {is_pct[-1]:.2f} ±{(is_pct_upper_ci[-1] - is_pct_lower_ci[-1]) / 2:.4f}% of total area')

    print(f'Adjusted IS area change from 1988 to 2020 with 95% CI')
    array_total_is_area_change_1988_2020 = df_is_change_sum_conus_plot_adjust['total_is_area_change_adjust'].values / 1e6  # convert to km2
    array_total_is_area_change_1988_2020_ci_upper = df_is_change_sum_conus_plot_adjust['total_is_area_change_ci_upper'].values / 1e6  # convert to km2
    array_total_is_area_change_1988_2020_ci_lower = df_is_change_sum_conus_plot_adjust['total_is_area_change_ci_lower'].values / 1e6  # convert to km2

    is_area_change_1988_2020 = np.nansum(array_total_is_area_change_1988_2020)  # convert to km2
    is_area_change_1988_2020_ci_upper = np.nansum(array_total_is_area_change_1988_2020_ci_upper)  # convert to km2
    is_area_change_1988_2020_ci_lower = np.nansum(array_total_is_area_change_1988_2020_ci_lower)  # convert to km2

    print(f'Change: {np.nansum(array_total_is_area_change_1988_2020):.0f}±{(is_area_change_1988_2020_ci_upper - is_area_change_1988_2020_ci_lower) / 2:.2f} km2  ')

    # average IS growth rate from 1988 to 2008
    mask_2005_2008 = (df_is_change_sum_conus_plot_adjust['year_1'] >= 2005) & (df_is_change_sum_conus_plot_adjust['year_1'] < 2008)
    mask_2008_2011 = (df_is_change_sum_conus_plot_adjust['year_1'] >= 2008) & (df_is_change_sum_conus_plot_adjust['year_1'] < 2011)
    mask_2017_2020 = (df_is_change_sum_conus_plot_adjust['year_1'] >= 2017) & (df_is_change_sum_conus_plot_adjust['year_1'] < 2020)

    average_total_is_area_increase_2005_2008 = np.nanmean(array_total_is_area_change_1988_2020[mask_2005_2008])  # in km2/year
    average_total_is_area_increase_2008_2011 = np.nanmean(array_total_is_area_change_1988_2020[mask_2008_2011])  # in km2/year

    average_total_is_area_increase_2005_2008_ci_upper = np.nanmean(array_total_is_area_change_1988_2020_ci_upper[mask_2005_2008])  # in km2/year
    average_total_is_area_increase_2005_2008_ci_lower = np.nanmean(array_total_is_area_change_1988_2020_ci_lower[mask_2005_2008])  # in km2/year

    average_total_is_area_increase_2008_2011_ci_upper = np.nanmean(array_total_is_area_change_1988_2020_ci_upper[mask_2008_2011])  # in km2/year
    average_total_is_area_increase_2008_2011_ci_lower = np.nanmean(array_total_is_area_change_1988_2020_ci_lower[mask_2008_2011])  # in km2/year

    print(f'Average total IS area increase from 2005 to 2008: {average_total_is_area_increase_2005_2008:.2f} km2/year')
    print(f'Average total IS area increase from 2005 to 2008 95% CI: ±{(average_total_is_area_increase_2005_2008_ci_upper - average_total_is_area_increase_2005_2008_ci_lower) / 2:.2f} km2/year')

    print(f'Average total IS area increase from 2008 to 2011: {average_total_is_area_increase_2008_2011:.2f} km2/year')
    print(f'Average total IS area increase from 2008 to 2011 95% CI: ±{(average_total_is_area_increase_2008_2011_ci_upper - average_total_is_area_increase_2008_2011_ci_lower) / 2:.2f} km2/year')

    is_rate_reduction = (average_total_is_area_increase_2008_2011 - average_total_is_area_increase_2005_2008) / average_total_is_area_increase_2005_2008
    is_rate_reduction_ci_upper = (average_total_is_area_increase_2008_2011_ci_upper - average_total_is_area_increase_2005_2008_ci_upper) / average_total_is_area_increase_2005_2008_ci_upper
    is_rate_reduction_ci_lower = (average_total_is_area_increase_2008_2011_ci_lower - average_total_is_area_increase_2005_2008_ci_lower) / average_total_is_area_increase_2005_2008_ci_lower

    print(f'Average total IS area increase reduction: {is_rate_reduction * 100:.4f} %')
    print(f'Average total IS area increase reduction 95% CI: ({is_rate_reduction_ci_lower * 100:.4f} %, {is_rate_reduction_ci_upper * 100:.4f} %)')

    average_area_is_area_increase_2017_2020 = np.nanmean(array_total_is_area_change_1988_2020[mask_2017_2020])

    average_area_is_area_increase_2017_2020_ci_upper = np.nanmean(array_total_is_area_change_1988_2020_ci_upper[mask_2017_2020])
    average_area_is_area_increase_2017_2020_ci_lower = np.nanmean(array_total_is_area_change_1988_2020_ci_lower[mask_2017_2020])  # in km2/year

    print(f'Average total IS area increase from 2017 to 2020: {average_area_is_area_increase_2017_2020:.4f} km2/year')
    print(f'Average total IS area increase from 2017 to 2020 95% CI: ±{(average_area_is_area_increase_2017_2020_ci_upper - average_area_is_area_increase_2017_2020_ci_lower) / 2:.2f} km2/year')

    is_rate_recovery = average_area_is_area_increase_2017_2020 / average_total_is_area_increase_2005_2008
    is_rate_recovery_ci_upper = average_area_is_area_increase_2017_2020_ci_upper / average_total_is_area_increase_2005_2008_ci_upper
    is_rate_recovery_ci_lower = average_area_is_area_increase_2017_2020_ci_lower / average_total_is_area_increase_2005_2008_ci_lower

    print(f'Average total IS area increase recovery: {is_rate_recovery * 100:.4f} %')
    print(f'Average total IS area increase recovery 95% CI: ({is_rate_recovery_ci_lower * 100:.4f} %, {is_rate_recovery_ci_upper * 100:.4f} %)')









