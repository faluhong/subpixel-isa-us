"""
    utility function to plot the IS area change for manuscript
"""

import numpy as np
import os
from os.path import join, exists
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

# from analysis.utils_isp_change_stats_analysis import (plot_isp_change_type_ts, plot_is_area_ts)


def plot_is_area_ts(df_is_change_sum,
                    title=None,
                    x_label='Year',
                    y_label='Area (km^2)',
                    y_label_right='Area percentage (%)',
                    x_axis_interval=None,
                    y_axis_interval=None,
                    flag_save=False,
                    output_filename=None,
                    axes=None,
                    right_decimals=2,
                    figsize=(20, 12),
                    xlim=None,
                    ylim=None,
                    plot_flag='area',
                    legend_flag=True,
                    flag_highlight_2008=False,
                    flag_adjust_with_ci=False,
                    flag_fill_95_ci=True,
                    ):
    """
        plot the IS area change trend for each year
        left y-axis: IS area (km^2); right y-axis: IS area percentage
    """

    # sns.set_theme()
    sns.set_style("white")

    if axes is None:
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    legend_size = 24
    tick_label_size = 28
    axis_label_size = 30
    title_size = 32
    tick_length = 4

    line_width = 3.0
    linestyle = 'solid'

    axes.set_title(title, fontsize=title_size)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(2)
        axes.spines[axis].set_linewidth(2)

    array_year_plot = np.concatenate([df_is_change_sum['year_1'].values, np.array([df_is_change_sum['year_2'].values[-1]])])

    if plot_flag == 'area':

        if flag_adjust_with_ci:
            # plot the adjusted IS area changes with confidence interval

            array_is_area = np.concatenate([df_is_change_sum['is_area_year_1_adjust'].values, np.array([df_is_change_sum['is_area_year_2_adjust'].values[-1]])])
            array_is_area = array_is_area / 1000000  # convert the area to km^2

            axes.plot(array_year_plot, array_is_area, label='Adjusted IS area', color='#363737',
                      marker='o', markersize=14,
                      linestyle=linestyle, linewidth=line_width)

            array_is_area_upper = np.concatenate([df_is_change_sum['is_area_year_1_ci_lower'].values,
                                                  np.array([df_is_change_sum['is_area_year_2_ci_lower'].values[-1]])]) / 1000000
            array_is_area_lower = np.concatenate([df_is_change_sum['is_area_year_1_ci_upper'].values,
                                                  np.array([df_is_change_sum['is_area_year_2_ci_upper'].values[-1]])]) / 1000000

            if flag_fill_95_ci:
                axes.fill_between(array_year_plot,
                                  array_is_area_upper,
                                  array_is_area_lower,
                                  color='#363737',
                                  alpha=0.2,
                                  )

        else:

            # plot the mapped IS area changes
            array_is_area = np.concatenate([df_is_change_sum['is_area_year_1'].values, np.array([df_is_change_sum['is_area_year_2'].values[-1]])])
            array_is_area = array_is_area / 1000000  # convert the area to km^2

            axes.plot(array_year_plot, array_is_area, label='IS area', color='#363737',
                      marker='o', markersize=14,
                      linestyle=linestyle, linewidth=line_width)

    elif plot_flag == 'count':

        area_is_count_year_1 = (df_is_change_sum['count_stable_is'].values +
                                df_is_change_sum['count_is_intensification'].values +
                                df_is_change_sum['count_is_decline'].values +
                                df_is_change_sum['count_is_reversal'].values) * 900

        area_is_count_year_2 = (df_is_change_sum['count_stable_is'].values +
                                df_is_change_sum['count_is_expansion'].values +
                                df_is_change_sum['count_is_intensification'].values +
                                df_is_change_sum['count_is_decline'].values) * 900

        area_is_count = np.concatenate([area_is_count_year_1, np.array([area_is_count_year_2[-1]])])
        area_is_count = area_is_count / 1000000  # convert the area to km^2

        axes.plot(array_year_plot, area_is_count, label='IS count area', color='#363737',
                  marker='o', markersize=14,
                  linestyle=linestyle, linewidth=line_width)

    if flag_highlight_2008:
        axes.axvline(x=2008, color='#f97306', linestyle='--', linewidth=line_width, label='2008')  # highlight the year 2008

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

    if x_axis_interval is not None:
        axes.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))
    if y_axis_interval is not None:
        axes.yaxis.set_major_locator(plticker.MultipleLocator(base=y_axis_interval))

    # Create a FuncFormatter to change the four-digit year to two-digit year
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x) % 100:02d}'))
    axes.ticklabel_format(style='plain', axis='y', useOffset=False)  # disable scientific notation

    axes.set_xlabel(x_label, size=axis_label_size)
    axes.set_ylabel(y_label, size=axis_label_size)

    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)

    # Remove duplicates by converting to a dictionary (preserves order)
    handles, labels = axes.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    if legend_flag:
        axes.legend(by_label.values(), by_label.keys(), loc='best', fontsize=legend_size)

    # add the second y-axis to show the percentage of the total area, the right y-axis
    ax_right = axes.secondary_yaxis('right')  # set the second y-axis, copy from the left y-axis
    ax_right.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    ax_right.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, right=True, which='major')

    ax_ticks = axes.get_yticks()  # get the ticks of the left y-axis
    ax_right_tick_labels = ax_ticks / (df_is_change_sum['total_area'].values[-1] / 1000000) * 100  # convert the area to percentage
    ax_right_tick_labels = np.round(ax_right_tick_labels, decimals=right_decimals)  # round the percentage to two decimal places

    # Use FixedLocator to ensure the tick labels are correctly aligned
    ax_right.yaxis.set_major_locator(FixedLocator(ax_ticks))
    ax_right.set_yticklabels(ax_right_tick_labels)  # set the right y-axis with the percentage label

    ax_right.set_ylabel(y_label_right, size=axis_label_size, labelpad=15)

    plt.tight_layout()

    if flag_save:
        assert output_filename is not None, 'output_filename is not provided'
        if not exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        plt.savefig(output_filename, dpi=300)
        plt.close()
    else:
        plt.show()



def plot_isp_change_type_ts(df_is_change_sum,
                            plot_flag,
                            title=None,
                            x_label='Year',
                            y_label='Count',
                            y_label_right='Percentage',
                            x_axis_interval=None,
                            y_axis_interval=None,
                            flag_save=False,
                            output_filename=None,
                            right_decimals=2,
                            axes=None,
                            figsize=(20, 12),
                            flag_plot_is_change_in_detail=True,
                            xlim=None,
                            ylim=None,
                            sns_style='white',
                            legend_flag=True,
                            flag_highlight_2008=False,
                            flag_adjust_with_ci=False,
                            fill_alpha=0.2,
                            flag_focus_on_growth_types=False,
                            flag_fill_95_ci=True,
                            ):
    """
    plot the trend of IS change type for each year

    :param df_is_change_sum: the dataframe of the ISP change summary, output from get_isp_change_summary_dataframe function
    :param plot_flag: flag to plot 'count' or 'area'
    :param title:
    :param x_label:
    :param y_label:
    :param y_label_right:
    :param x_axis_interval:
    :param y_axis_interval:
    :param flag_save:
    :param output_filename:
    :param axes:
    :param right_decimals:
    :param flag_focus_on_growth_types: flag to only focus on IS expansion and IS intensification, default is False, i.e., plot all IS change types
    :return:
    """

    # sns.set_theme()
    sns.set_style(sns_style)

    if axes is None:
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    legend_size = 24
    tick_label_size = 28
    axis_label_size = 30
    title_size = 32
    tick_length = 4

    line_width = 3.0
    linestyle = 'solid'

    axes.set_title(title, fontsize=title_size)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(2)
        axes.spines[axis].set_linewidth(2)

    array_year_plot = df_is_change_sum['year_2'].values

    if plot_flag == 'count':
        array_is_expansion = df_is_change_sum['count_is_expansion'].values * 900 / 1000000  # convert the pixel count to area in km^2
        array_is_intensification = df_is_change_sum['count_is_intensification'].values * 900 / 1000000
        array_is_declination = df_is_change_sum['count_is_decline'].values * 900 / 1000000
        array_is_reversal = df_is_change_sum['count_is_reversal'].values * 900 / 1000000

        array_is_total_area_change = array_is_expansion - array_is_reversal

    elif plot_flag == 'area':

        if flag_adjust_with_ci:

            array_is_expansion = df_is_change_sum['area_is_expansion_adjust'].values / 1000000
            array_is_intensification = df_is_change_sum['area_is_intensification_adjust'].values / 1000000
            array_is_declination = df_is_change_sum['area_is_decline_adjust'].values / 1000000
            array_is_reversal = df_is_change_sum['area_is_reversal_adjust'].values / 1000000

            array_is_total_area_change = df_is_change_sum['total_is_area_change_adjust'].values / 1000000

            array_is_expansion_upper = df_is_change_sum['area_is_expansion_ci_upper'].values / 1000000
            array_is_expansion_lower = df_is_change_sum['area_is_expansion_ci_lower'].values / 1000000

            array_is_intensification_upper = df_is_change_sum['area_is_intensification_ci_upper'].values / 1000000
            array_is_intensification_lower = df_is_change_sum['area_is_intensification_ci_lower'].values / 1000000

            array_is_decline_upper = df_is_change_sum['area_is_decline_ci_upper'].values / 1000000
            array_is_decline_lower = df_is_change_sum['area_is_decline_ci_lower'].values / 1000000

            array_is_reversal_upper = df_is_change_sum['area_is_reversal_ci_upper'].values / 1000000
            array_is_reversal_lower = df_is_change_sum['area_is_reversal_ci_lower'].values / 1000000

            array_is_total_area_change_upper = df_is_change_sum['total_is_area_change_ci_upper'].values / 1000000
            array_is_total_area_change_lower = df_is_change_sum['total_is_area_change_ci_lower'].values / 1000000
        else:

            # convert the unit from m^2 to km^2
            array_is_expansion = df_is_change_sum['area_is_expansion'].values / 1000000
            array_is_intensification = df_is_change_sum['area_is_intensification'].values / 1000000
            array_is_declination = df_is_change_sum['area_is_decline'].values / 1000000
            array_is_reversal = df_is_change_sum['area_is_reversal'].values / 1000000

            array_is_total_area_change = df_is_change_sum['total_is_area_change'].values / 1000000

    else:
        raise ValueError('The plot_flag is not correct')

    if flag_plot_is_change_in_detail:

        if flag_adjust_with_ci:
            axes.plot(array_year_plot, array_is_expansion, label='IS expansion', color='#e50000',
                      marker='o', markersize=12,
                      linestyle=linestyle, linewidth=line_width)

            axes.plot(array_year_plot, array_is_intensification, label='IS intensification', color='#9f28eb',
                      marker='s', markersize=11,
                      linestyle=linestyle, linewidth=line_width)

            if flag_fill_95_ci:
                axes.fill_between(array_year_plot,
                                  array_is_expansion_upper,
                                  array_is_expansion_lower,
                                  color='#e50000',
                                  alpha=fill_alpha,)

                axes.fill_between(array_year_plot,
                                  array_is_intensification_upper,
                                  array_is_intensification_lower,
                                  color='#9f28eb',
                                  alpha=fill_alpha,)

                axes.fill_between(array_year_plot,
                                  array_is_total_area_change_upper,
                                  array_is_total_area_change_lower,
                                  color='#363737',
                                  alpha=fill_alpha, )

            if flag_focus_on_growth_types is False:

                axes.plot(array_year_plot, array_is_declination, label='IS decline', color='#0485d1',
                          marker='D', markersize=11,
                          linestyle=linestyle, linewidth=line_width)

                axes.plot(array_year_plot, array_is_reversal, label='IS reversal', color='#02ab2e',
                          marker='^', markersize=14,
                          linestyle=linestyle, linewidth=line_width)

                if flag_fill_95_ci:
                    axes.fill_between(array_year_plot,
                                      array_is_decline_upper,
                                      array_is_decline_lower,
                                      color='#0485d1',
                                      alpha=fill_alpha,)

                    axes.fill_between(array_year_plot,
                                      array_is_reversal_upper,
                                      array_is_reversal_lower,
                                      color='#02ab2e',
                                      alpha=fill_alpha, )
        else:

            axes.plot(array_year_plot, array_is_expansion, label='IS expansion', color='#e50000',
                      marker='o', markersize=12,
                      linestyle=linestyle, linewidth=line_width)

            axes.plot(array_year_plot, array_is_intensification, label='IS intensification', color='#9f28eb',
                      marker='s', markersize=11,
                      linestyle=linestyle, linewidth=line_width)

            if flag_focus_on_growth_types is False:

                axes.plot(array_year_plot, array_is_declination, label='IS decline', color='#0485d1',
                          marker='D', markersize=11,
                          linestyle=linestyle, linewidth=line_width)

                axes.plot(array_year_plot, array_is_reversal, label='IS reversal', color='#02ab2e',
                          marker='^', markersize=14,
                          linestyle=linestyle, linewidth=line_width)

    if flag_highlight_2008:
        axes.axvline(x=2008, color='#f97306', linestyle='--', linewidth=line_width, label='2008')  # highlight the year 2008

    # if plot_flag == 'area':
    axes.plot(array_year_plot, array_is_total_area_change, label='Total IS area change', color='#363737',
              marker='X', markersize=14,
              linestyle=linestyle, linewidth=line_width)

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

    if x_axis_interval is not None:
        axes.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))
    if y_axis_interval is not None:
        axes.yaxis.set_major_locator(plticker.MultipleLocator(base=y_axis_interval))

    # Create a FuncFormatter to change the four-digit year to two-digit year
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x) % 100:02d}'))
    axes.ticklabel_format(style='plain', axis='y', useOffset=False)  # disable scientific notation

    axes.set_xlabel(x_label, size=axis_label_size)
    axes.set_ylabel(y_label, size=axis_label_size)

    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)

    if legend_flag:
        # Remove duplicates by converting to a dictionary (preserves order)
        handles, labels = axes.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        axes.legend(by_label.values(), by_label.keys(), loc='best', fontsize=legend_size)

    # add the second y-axis to show the percentage of the total area, the right y-axis
    ax_right = axes.secondary_yaxis('right')  # set the second y-axis, copy from the left y-axis
    ax_right.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    ax_right.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, right=True, which='major')

    ax_ticks = axes.get_yticks()  # get the ticks of the left y-axis

    if plot_flag == 'area':
        ax_right_tick_labels = ax_ticks / (df_is_change_sum['total_area'].values[-1] / 1000000) * 100  # convert the area to percentage
    elif plot_flag == 'count':
        ax_right_tick_labels = ax_ticks / (df_is_change_sum['total_area'].values[-1] / 1000000) * 100
    else:
        raise ValueError('The plot_flag is not correct')

    ax_right_tick_labels = np.round(ax_right_tick_labels, decimals=right_decimals)  # round the percentage to decimal places

    # Use FixedLocator to ensure the tick labels are correctly aligned
    ax_right.yaxis.set_major_locator(FixedLocator(ax_ticks))
    ax_right.set_yticklabels(ax_right_tick_labels)  # set the right y-axis with the percentage label

    ax_right.set_ylabel(y_label_right, size=axis_label_size, labelpad=15)

    plt.tight_layout()

    if flag_save:
        assert output_filename is not None, 'output_filename is not provided'
        if not exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        plt.savefig(output_filename, dpi=300)
        plt.close()
    else:
        plt.show()


def manuscript_is_area_plot(df_is_change_sum_conus_plot, title_1, title_2, figsize=(24, 18),
                            xlim=None, ylim_1=None, ylim_2=None,
                            output_flag=False, output_filename=None,
                            sns_style='white',
                            legend_flag=True,
                            plot_flag='area',
                            flag_highlight_2008=False,
                            flag_adjust_with_ci=False,
                            fill_alpha=0.2,
                            flag_focus_on_growth_types=False,
                            flag_fill_95_ci=True,):
    """
        plot the IS area change for manuscript.
        Left figure is the IS area time series plot.
        Right figure is the IS area change type time series plot.

        :param df_is_change_sum_conus_plot:
        :param title_1:
        :param title_2:
        :param figsize:
        :param xlim:
        :param ylim_1:
        :param ylim_2:
        :param output_flag:
        :param output_filename:
        :param sns_style:
        :param legend_flag:
        :param plot_flag:
        :param flag_highlight_2008:
        :return:
    """

    sns.set_style(sns_style)
    figure_twin, axes_twin = plt.subplots(ncols=2, nrows=1, figsize=figsize)

    ax = plt.subplot(1, 2, 1)

    plot_is_area_ts(df_is_change_sum=df_is_change_sum_conus_plot,
                    title=title_1,
                    x_label='Year',
                    y_label='Area (km$^2$)',
                    y_label_right='Area percentage (%)',
                    x_axis_interval=2,
                    y_axis_interval=None,
                    flag_save=False,
                    output_filename=None,
                    axes=ax,
                    right_decimals=3,
                    figsize=(18, 10),
                    xlim=xlim,
                    ylim=ylim_1,
                    plot_flag=plot_flag,
                    legend_flag=legend_flag,
                    flag_highlight_2008=flag_highlight_2008,
                    flag_adjust_with_ci=flag_adjust_with_ci,
                    flag_fill_95_ci=flag_fill_95_ci,
                    )

    ax = plt.subplot(1, 2, 2)

    if xlim is None:
        x_lim_isp_change = None
    else:
        x_lim_isp_change = (xlim[0] + 1, xlim[-1])

    plot_isp_change_type_ts(df_is_change_sum_conus_plot,
                            plot_flag=plot_flag,
                            title=title_2,
                            x_label='Year',
                            y_label='Area (km$^2$)',
                            y_label_right='Area percentage (%)',
                            x_axis_interval=2,
                            y_axis_interval=None,
                            flag_save=False,
                            output_filename=None,
                            axes=ax,
                            right_decimals=3,
                            figsize=(18, 10),
                            xlim=x_lim_isp_change,
                            ylim=ylim_2,
                            sns_style=sns_style,
                            legend_flag=legend_flag,
                            flag_highlight_2008=flag_highlight_2008,
                            flag_adjust_with_ci=flag_adjust_with_ci,
                            fill_alpha=fill_alpha,
                            flag_focus_on_growth_types=flag_focus_on_growth_types,
                            flag_fill_95_ci=flag_fill_95_ci,
                            )

    plt.tight_layout()

    if output_flag:
        assert output_filename is not None, 'output_filename is not provided'
        if not exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        plt.savefig(output_filename, dpi=300)
        plt.close()
    else:
        plt.show()

