"""
    utility functions to prepare the annual state/MSA IS, GDP, and population data for analysis
"""

import numpy as np
import os
from os.path import join
import sys
import pandas as pd
import geopandas as gpd

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from pythoncode.conus_isa_centroid.is_centroid_extraction import (get_target_basic_info)


def read_state_is_area(df_conus_state_basic_info, data_flag, isp_folder,
                       array_target_year=None,
                       flag_adjust=False):
    """
    read the state-level ISP data, including the ISP area and percentage

    :param df_conus_state_basic_info:
    :param data_flag:
    :param isp_folder:
    :param array_target_year:
    :param flag_adjust: flag to indicate whether to use the adjusted IS area
    :return:
    """

    if data_flag == 'conus_isp':
        array_isp_year = np.arange(1985, 2023, 1)
    elif data_flag == 'annual_nlcd':
        array_isp_year = np.arange(1985, 2024, 1)
    else:
        raise ValueError('The data_flag is not correct')

    array_is_pct = np.zeros((len(df_conus_state_basic_info), len(array_isp_year)))
    array_is_area = np.zeros((len(df_conus_state_basic_info), len(array_isp_year)))

    if flag_adjust is False:
        # read the map-based IS area
        filename_extract_state_is_area = join(rootpath, 'results', 'isp_change_stats', 'state_level', data_flag,
                                              f'state_isp_stats_{isp_folder}.csv')

        df_state_isp_annual_change = pd.read_csv(filename_extract_state_is_area)

        for i_state in range(0, len(df_conus_state_basic_info)):
            array_is_pct[i_state, :] = df_state_isp_annual_change.iloc[i_state, 3:3 + len(array_isp_year)].values
            array_is_area[i_state, :] = df_state_isp_annual_change.iloc[i_state, 4 + len(array_isp_year):].values

    else:
        filename_extract_state_is_area = join(rootpath, 'results', 'isp_change_stats', 'state_level', data_flag,
                                              f'conus_state_is_stats_adjust_{isp_folder}.gpkg')
        df_state_isp_annual_change = gpd.read_file(filename_extract_state_is_area)

        # drop the geometry column
        df_state_isp_annual_change = pd.DataFrame(df_state_isp_annual_change.drop(columns=['geometry', 'total_area']))

        for i_state in range(0, len(df_conus_state_basic_info)):
            for i_year in range(0, len(array_isp_year)):
                array_is_pct[i_state, i_year] = df_state_isp_annual_change.loc[i_state, f'is_pct_{array_isp_year[i_year]}']
                array_is_area[i_state, i_year] = df_state_isp_annual_change.loc[i_state, f'is_area_{array_isp_year[i_year]}']

        array_is_area = array_is_area / 1e6     # convert from m2 to km2

    if array_target_year is not None:
        array_return_year = array_isp_year[np.isin(array_isp_year, array_target_year)]
        array_return_is_area = array_is_area[:, np.isin(array_isp_year, array_target_year)]
        array_return_is_pct = array_is_pct[:, np.isin(array_isp_year, array_target_year)]
    else:
        array_return_year = array_isp_year
        array_return_is_area = array_is_area
        array_return_is_pct = array_is_pct

    return (array_return_year, array_return_is_area, array_return_is_pct)


def extract_state_gdp(df_conus_state_basic_info, df_state_gdp, array_target_year=None):
    """
        extract the state annual GDP data, including the real GDP and nominal GDP
        Args:
            df_conus_state_basic_info: the basic info of the CONUS state, including the state name, id, etc.
            path_urban_pulse: the path of the urban pulse data

        Returns:
    """

    array_gdp_year = np.arange(1997, 2023)
    # array_target_year = np.arange(1998, 2020, 1)  # the target year to extract the ISP, 2019-2020 change is included

    array_real_gdp = np.zeros((len(df_conus_state_basic_info), len(array_gdp_year)))
    array_nominal_gdp = np.zeros((len(df_conus_state_basic_info), len(array_gdp_year)))

    for i_state in range(0, len(df_conus_state_basic_info)):
    # for i_state in range(10, 11):

        state_name = df_conus_state_basic_info['NAME'].values[i_state]

        df_state_gdp_single = df_state_gdp[df_state_gdp['GeoName'] == state_name]

        array_real_gdp[i_state, :] = df_state_gdp_single.iloc[0, 8:8 + len(array_gdp_year)].values
        array_nominal_gdp[i_state, :] = df_state_gdp_single.iloc[2, 8:8 + len(array_gdp_year)].values

    # convert from million $ to billion $
    array_real_gdp = array_real_gdp / 1000
    array_nominal_gdp = array_nominal_gdp / 1000

    if array_target_year is not None:
        array_return_year = array_gdp_year[np.isin(array_gdp_year, array_target_year)]
        array_return_real_gdp = array_real_gdp[:, np.isin(array_gdp_year, array_target_year)]
        array_return_nominal_gdp = array_nominal_gdp[:, np.isin(array_gdp_year, array_target_year)]
    else:
        array_return_year = array_gdp_year
        array_return_real_gdp = array_real_gdp
        array_return_nominal_gdp = array_nominal_gdp

    return (array_return_year, array_return_real_gdp, array_return_nominal_gdp)


def extract_state_population(df_conus_state_basic_info, df_state_pop, array_target_year=None):
    """
        extract the state population data
        :param
            df_conus_state_basic_info: the basic info of the CONUS state, including the state name, id, etc.
            df_state_pop: dataframe containing the population data
        :returns
    """

    array_pop_year = np.arange(1929, 2024)
    array_pop = np.zeros((len(df_conus_state_basic_info), len(array_pop_year)))

    for i_state in range(0, len(df_conus_state_basic_info)):
    # for i_state in range(10, 11):

        state_name = df_conus_state_basic_info['NAME'].values[i_state]
        # print(state_name)

        df_state_pop_single = df_state_pop[df_state_pop['GeoName'] == state_name]
        array_pop[i_state, :] = df_state_pop_single.iloc[1, 8:8 + len(array_pop_year)].values

    if array_target_year is not None:
        array_return_year = array_pop_year[np.isin(array_pop_year, array_target_year)]
        array_return_pop = array_pop[:, np.isin(array_pop_year, array_target_year)]
    else:
        array_return_year = array_pop_year
        array_return_pop = array_pop

    array_return_pop = array_return_pop / 1000  # convert the population to thousands

    return (array_return_year, array_return_pop)


def prepare_ard_state_is_gdp_population(path_urban_pulse, data_flag, isp_folder, array_target_year,
                                        flag_adjust=False):
    """
        prepare the annual state IS, GDP, and population data for analysis
    """

    df_conus_state_basic_info = get_target_basic_info(modify_target='state',
                                                      flag_keep_geometry=True,
                                                      flag_wgs84=True)

    # read the annual state GDP data, GDP data range from 1997 to 2023
    filename_summary = join(path_urban_pulse, 'state_level', 'SAGDP', 'SAGDP1__ALL_AREAS_1997_2023.csv')
    df_state_gdp = pd.read_csv(filename_summary)

    # read the annual state population data, population data range from 1929 to 2023
    filename_state_pop_summary = join(path_urban_pulse, 'state_level', 'SAINC', 'SAINC1__ALL_AREAS_1929_2023.csv')
    df_state_pop = pd.read_csv(filename_state_pop_summary, encoding='latin1')

    # keep the analysis period to from 1997 to 2020.
    # Because: (1) GDP data is available from 1997 to 2023, (2) Good-quality ISP data is available from 1988 to 2020

    (array_return_year,
     array_real_gdp,
     array_nominal_gdp) = extract_state_gdp(df_conus_state_basic_info,
                                            df_state_gdp,
                                            array_target_year=array_target_year)

    (array_return_year, array_pop) = extract_state_population(df_conus_state_basic_info,
                                                              df_state_pop,
                                                              array_target_year=array_target_year)

    (array_return_year,
     array_is_area,
     array_is_pct) = read_state_is_area(df_conus_state_basic_info,
                                        data_flag,
                                        isp_folder,
                                        array_target_year=array_target_year,
                                        flag_adjust=flag_adjust,)

    return (df_conus_state_basic_info, array_is_area, array_is_pct, array_real_gdp, array_nominal_gdp, array_pop)


def read_msa_is_area(df_conus_msa_basic_info, data_flag, isp_folder,
                     array_target_year=None,
                     flag_adjust=False,):
    """
    read the county level ISP data, including the ISP area and percentage

    :param df_conus_msa_basic_info:
    :param data_flag:
    :param isp_folder:
    :param array_target_year:
    :return:
    """

    if data_flag == 'conus_isp':
        array_isp_year = np.arange(1985, 2023, 1)
    elif data_flag == 'annual_nlcd':
        array_isp_year = np.arange(1985, 2024, 1)
    else:
        raise ValueError('The data_flag is not correct')

    array_is_pct = np.zeros((len(df_conus_msa_basic_info), len(array_isp_year)))
    array_is_area = np.zeros((len(df_conus_msa_basic_info), len(array_isp_year)))

    if flag_adjust is False:
        # read the map-based IS area
        filename_extract_county_is_area = join(rootpath, 'results', 'isp_change_stats', 'msa_level', data_flag,
                                               f'msa_isp_stats_{isp_folder}.csv')
        df_msa_isp_annual_change = pd.read_csv(filename_extract_county_is_area)

        for i_obj in range(0, len(df_conus_msa_basic_info)):
            # use CBSAFP column to match
            string_geofips = int(df_conus_msa_basic_info['CBSAFP'].values[i_obj])
            msa_name = df_conus_msa_basic_info['NAME'].values[i_obj]

            mask_match = df_msa_isp_annual_change['CBSAFP'].values == string_geofips
            assert np.count_nonzero(mask_match) == 1, f'no match for {string_geofips} {msa_name}'

            array_is_pct[i_obj, :] = df_msa_isp_annual_change[mask_match].iloc[0, 9:9 + len(array_isp_year)].values
            array_is_area[i_obj, :] = df_msa_isp_annual_change[mask_match].iloc[0, 10 + len(array_isp_year):].values

    else:
        ##
        filename_extract_msa_is_area = join(rootpath, 'results', 'isp_change_stats', 'msa_level', data_flag,
                                            f'conus_msa_is_stats_adjust_{isp_folder}.gpkg')
        df_msa_isp_annual_change = gpd.read_file(filename_extract_msa_is_area)

        # drop the geometry column
        df_msa_isp_annual_change = pd.DataFrame(df_msa_isp_annual_change.drop(columns=['geometry', 'total_area']))

        for i_obj in range(0, len(df_conus_msa_basic_info)):

            string_geofips = df_conus_msa_basic_info['CBSAFP'].values[i_obj]
            msa_name = df_conus_msa_basic_info['NAME'].values[i_obj]

            mask_match = df_msa_isp_annual_change['CBSAFP'].values == string_geofips
            assert np.count_nonzero(mask_match) == 1, f'no match for {string_geofips} {msa_name}'

            for i_year in range(0, len(array_isp_year)):
                array_is_pct[i_obj, i_year] = df_msa_isp_annual_change[mask_match][f'is_pct_{array_isp_year[i_year]}'].values[0]
                array_is_area[i_obj, i_year] = df_msa_isp_annual_change[mask_match][f'is_area_{array_isp_year[i_year]}'].values[0]

        array_is_area = array_is_area / 1e6     # convert from m2 to km2

    if array_target_year is not None:
        array_return_year = array_isp_year[np.isin(array_isp_year, array_target_year)]
        array_return_is_area = array_is_area[:, np.isin(array_isp_year, array_target_year)]
        array_return_is_pct = array_is_pct[:, np.isin(array_isp_year, array_target_year)]
    else:
        array_return_year = array_isp_year
        array_return_is_area = array_is_area
        array_return_is_pct = array_is_pct

    return (array_return_year, array_return_is_area, array_return_is_pct)


def extract_msa_gdp(df_conus_msa_basic_info, df_msa_gdp, array_target_year=None):
    """
        extract the state annual GDP data, including the real GDP and nominal GDP
        Args:
            df_conus_state_basic_info: the basic info of the CONUS state, including the state name, id, etc.
            path_urban_pulse: the path of the urban pulse data

        Returns:
    """

    array_gdp_year = np.arange(2001, 2023)

    array_real_gdp = np.zeros((len(df_conus_msa_basic_info), len(array_gdp_year)))
    array_nominal_gdp = np.zeros((len(df_conus_msa_basic_info), len(array_gdp_year)))

    for i_obj in range(0, len(df_conus_msa_basic_info)):
        # for i_obj in range(766, 767):

        string_geofips = df_conus_msa_basic_info['CBSAFP'].values[i_obj]
        string_geofips = f'{string_geofips}'
        # msa_name = df_conus_msa_basic_info['NAME'].values[i_obj]

        mask_match = df_msa_gdp['GeoFips'].values == string_geofips

        if np.count_nonzero(mask_match) == 0:
            # print(f'no match for {string_geofips} {msa_name} ')
            array_real_gdp[i_obj, :] = np.nan
            continue

        list_real_gdp = df_msa_gdp[mask_match].iloc[0, 4:4 + len(array_gdp_year)].values
        list_nominal_gdp = df_msa_gdp[mask_match].iloc[2, 4:4 + len(array_gdp_year)].values

        # convert the string to float, and convert the '(NA)' to nan
        array_real_gdp_county = np.array([float(x) if x != '(NA)' else np.nan for x in list_real_gdp])
        array_nominal_gdp_county = np.array([float(x) if x != '(NA)' else np.nan for x in list_nominal_gdp])

        array_real_gdp[i_obj, :] = array_real_gdp_county
        array_nominal_gdp[i_obj, :] = array_nominal_gdp_county

    # convert from thousands $ to billions $
    array_real_gdp = array_real_gdp / 1000 / 1000
    array_nominal_gdp = array_nominal_gdp / 1000 / 1000

    if array_target_year is not None:
        array_return_year = array_gdp_year[np.isin(array_gdp_year, array_target_year)]
        array_return_real_gdp = array_real_gdp[:, np.isin(array_gdp_year, array_target_year)]
        array_return_nominal_gdp = array_nominal_gdp[:, np.isin(array_gdp_year, array_target_year)]
    else:
        array_return_year = array_gdp_year
        array_return_real_gdp = array_real_gdp
        array_return_nominal_gdp = array_nominal_gdp

    return (array_return_year, array_return_real_gdp, array_return_nominal_gdp)


def extract_msa_population(df_conus_msa_basic_info, df_msa_pop, array_target_year=None):
    """
        extract the state population data
        :param
            df_conus_state_basic_info: the basic info of the CONUS state, including the state name, id, etc.
            df_state_pop: dataframe containing the population data
        :returns
    """

    array_pop_year = np.arange(1969, 2023)

    array_pop = np.zeros((len(df_conus_msa_basic_info), len(array_pop_year)))

    for i_obj in range(0, len(df_conus_msa_basic_info)):

        string_geofips = df_conus_msa_basic_info['CBSAFP'].values[i_obj]
        string_geofips = f'{string_geofips}'
        msa_name = df_conus_msa_basic_info['NAME'].values[i_obj]

        mask_match = df_msa_pop['GeoFips'].values == string_geofips

        if np.count_nonzero(mask_match) == 0:
            # print(f'no match for {string_geofips} {msa_name}')
            array_pop[i_obj, :] = np.nan
            continue

        list_pop = df_msa_pop[mask_match].iloc[0, 2:2 + len(array_pop_year)].values

        array_pop_county = np.array([float(x) if x != '(NA)' else np.nan for x in list_pop])
        array_pop[i_obj, :] = array_pop_county

    if array_target_year is not None:
        array_return_year = array_pop_year[np.isin(array_pop_year, array_target_year)]
        array_return_pop = array_pop[:, np.isin(array_pop_year, array_target_year)]
    else:
        array_return_year = array_pop_year
        array_return_pop = array_pop

    array_return_pop = array_return_pop / 1000  # convert the population to thousands

    return (array_return_year, array_return_pop)


def prepare_ard_msa_is_gdp_population(path_urban_pulse, data_flag, isp_folder, array_target_year,
                                      flag_adjust=False):
    """
        prepare the annual MSA IS, GDP, and population data for analysis
        :param path_urban_pulse:
        :param data_flag:
        :param isp_folder:
        :param array_target_year:
        :return:
    """

    df_msa_basic_info = get_target_basic_info(modify_target='msa',
                                              flag_keep_geometry=True,
                                              flag_wgs84=True,
                                              flag_all_msa=False)

    # read the annual state GDP data, GDP data range from 2001 to 2023
    filename_msa_gdp_summary = join(path_urban_pulse, 'MSA_level', 'MSA_GDP.csv')
    df_msa_gdp = pd.read_csv(filename_msa_gdp_summary, encoding='latin1', skiprows=5)

    # read the annual state population data, population data range from 1929 to 2023
    filename_msa_pop_summary = join(path_urban_pulse, 'MSA_level', 'MSA_population.csv')
    df_msa_pop = pd.read_csv(filename_msa_pop_summary, encoding='latin1', skiprows=5)

    (array_return_year,
     array_real_gdp,
     array_nominal_gdp) = extract_msa_gdp(df_msa_basic_info,
                                          df_msa_gdp,
                                          array_target_year=array_target_year)

    (array_return_year, array_pop) = extract_msa_population(df_msa_basic_info,
                                                            df_msa_pop,
                                                            array_target_year=array_target_year)

    (array_return_year,
     array_is_area,
     array_is_pct) = read_msa_is_area(df_msa_basic_info,
                                      data_flag,
                                      isp_folder,
                                      array_target_year=array_target_year,
                                      flag_adjust=flag_adjust)

    # keep the row that does not contain nan value
    # mask_nan = np.isnan(array_real_gdp).all(axis=1) | np.isnan(array_nominal_gdp).all(axis=1) | np.isnan(array_pop).all(axis=1) | np.isnan(array_is_area).all(axis=1)

    # remove rows with any NaN values
    mask_nan = np.isnan(array_real_gdp).any(axis=1) | np.isnan(array_nominal_gdp).any(axis=1) | np.isnan(array_pop).any(axis=1) | np.isnan(array_is_area).any(axis=1)

    array_real_gdp = array_real_gdp[~mask_nan, :]
    array_nominal_gdp = array_nominal_gdp[~mask_nan, :]
    array_pop = array_pop[~mask_nan, :]
    array_is_area = array_is_area[~mask_nan, :]
    array_is_pct = array_is_pct[~mask_nan, :]

    df_msa_basic_info = df_msa_basic_info[~mask_nan]
    df_msa_basic_info = df_msa_basic_info.reset_index(drop=True)    # reset the index


    return (df_msa_basic_info, array_is_area, array_is_pct, array_real_gdp, array_nominal_gdp, array_pop)









