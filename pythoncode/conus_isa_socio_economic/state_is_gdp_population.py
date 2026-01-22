"""
    extract the annual state GDP and compare with ISP at the state level
"""

import numpy as np
import os
from os.path import join
import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.conus_isa_socio_economic.utils_plot_state_is_gdp import (sum_plot_is_area_gdp_change_overtime,)

from pythoncode.conus_isa_socio_economic.utils_prepare_ard_is_gdp_pop import (prepare_ard_state_is_gdp_population)


# def main():
if __name__ =='__main__':

    path_urban_pulse = join(rootpath, 'data', 'socio_economic')

    data_flag = 'conus_isp' # 'conus_isp' or 'annual_nlcd'
    isp_folder = 'individual_year_tile_post_processing_binary_is_ndvi015_sm'

    # keep the analysis period to from 1997 to 2020.
    # Because: (1) GDP data is available from 1997 to 2023, (2) Good-quality ISP data is available from 1988 to 2020
    array_target_year = np.arange(1997, 2021)

    print(data_flag, isp_folder)

    (df_conus_state_basic_info,
     array_is_area,
     array_is_pct,
     array_real_gdp,
     array_nominal_gdp,
     array_pop) = prepare_ard_state_is_gdp_population(path_urban_pulse,
                                                      data_flag,
                                                      isp_folder,
                                                      array_target_year,
                                                      flag_adjust=True)

    ##
    # plot the relationship between IS area and real GDP over time
    sum_plot_is_area_gdp_change_overtime(df_conus_state_basic_info=df_conus_state_basic_info,
                                         array_x_data=array_is_area,
                                         array_y_data=array_real_gdp,
                                         array_target_year=array_target_year,
                                         flag_plot_vline=True,
                                         super_x_label=f'IS area (km$^2$)',
                                         super_y_label='real GDP (billion $)',
                                         x_scale='linear',
                                         y_scale='linear',
                                         cmap='viridis',
                                         )








