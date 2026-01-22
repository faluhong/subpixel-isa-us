"""
    for each MSA, plot the IS area change and GDP change over time (2000-2019)
"""

import os
from os.path import join
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.conus_isa_socio_economic.utils_plot_state_is_gdp import plot_is_area_gdp_change_overtime
from pythoncode.conus_isa_socio_economic.utils_prepare_ard_is_gdp_pop import (prepare_ard_msa_is_gdp_population,)


# def main():
if __name__ =='__main__':

    path_urban_pulse = join(rootpath, 'data', 'socio_economic')

    # define the data flag and ISP folder, Annual NLCD ISP ranges from 1985 to 2023, CONUS ISP ranges from 1988 to 2020
    data_flag = 'conus_isp'  # 'conus_isp' or 'annual_nlcd'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    modify_target = 'msa'

    # define target years, GDP starts from 2001
    array_target_year_msa = np.arange(2001, 2021)

    (df_msa_basic_info,
     array_is_area_msa,
     array_is_pct,
     array_real_gdp_msa,
     array_nominal_gdp_msa,
     array_pop_msa) = prepare_ard_msa_is_gdp_population(path_urban_pulse,
                                                    data_flag,
                                                    isp_folder,
                                                    array_target_year_msa)

    # sort the state by the name alphabetically
    df_msa_basic_info_sorted = df_msa_basic_info.sort_values(by='NAME')

    ##
    flag_plot_vline = True
    super_x_label = f'IS area (km$^2$)'
    super_y_label = 'real GDP (billion $)'

    plt.rcParams['font.family'] = "Arial"

    for i_target_interval in range(0, len(df_msa_basic_info_sorted), 49):
    # for i_target_interval in range(0, 50, 49):

        if i_target_interval+49 > len(df_msa_basic_info_sorted):
            df_msa_basic_info_sorted_plot = df_msa_basic_info_sorted.iloc[i_target_interval:, :]
        else:
            df_msa_basic_info_sorted_plot = df_msa_basic_info_sorted.iloc[i_target_interval:i_target_interval+49, :]

        fig, ax_plot = plt.subplots(ncols=7, nrows=7, figsize=(32, 22.5))

        for i_target in range(0, len(df_msa_basic_info_sorted_plot)):
            region_name = df_msa_basic_info_sorted_plot['NAME'].values[i_target]
            print(f'Plotting MSA: {region_name} ({i_target + 1 + i_target_interval}/{len(df_msa_basic_info)})')

            index = df_msa_basic_info_sorted_plot.index[i_target]

            array_x_plot = array_is_area_msa[index, :]
            array_y_plot = array_real_gdp_msa[index, :]

            plot_is_area_gdp_change_overtime(array_x_plot,
                                             array_y_plot,
                                             array_target_year=array_target_year_msa,
                                             region_name=region_name,
                                             cmap='viridis',
                                             ax_plot=ax_plot[int(i_target / 7), i_target % 7],
                                             # x_label=f'IS area (km$^2$)',
                                             # y_label='real GDP (billion $)',
                                             x_label=None,
                                             y_label=None,
                                             cbar_flag=False,
                                             tick_label_size=18,
                                             axis_label_size=16,
                                             title_size=20,
                                             scatter_size=50,
                                             tick_length=4,
                                             cbar_tick_label_size=20,
                                             figsize=(14, 8),
                                             vline_width=2.0,
                                             flag_plot_vline=True,
                                             x_scale='linear',
                                             y_scale='linear',
                                             title=region_name)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)

        fig.supxlabel(super_x_label, fontsize=20)
        fig.supylabel(super_y_label, fontsize=20)

        # plt.savefig(join(rootpath, 'results', 'conus_is_socio_economic', f'{modify_target}_level',
        #                  f'msa_is_gdp_change_overtime_{i_target_interval+1}_{i_target_interval+50}.jpg'), dpi=300)
        # plt.close()
        # plt.show()













