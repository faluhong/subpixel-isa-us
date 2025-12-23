"""
    Good practice function for the accuracy assessment
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from lulc_validation.lulc_val import StratVal


def generate_good_practice_matrix(data, array_weight, array_count, confidence_interval=1.96):
    """
        generate the error matrix following the "Good Practice" strategy
        The validation sample design follows the stratified random sample

        Steps:
        (1) calculate the proportion of area for each cell
        (2) calculate the user's and producer's accuracy
        (3) calculate the variance of user's, producer's and overall accuracy
        (4) reorder the output dataframe

        Ref: Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014).
        Good practices for estimating area and assessing accuracy of land change. Remote sensing of Environment, 148, 42-57.
        https://www.sciencedirect.com/science/article/pii/S0034425714000704

        Args:
            data: 2-D array, the error matrix of sample counts
                  ROW represents the map class, COLUMN represents the reference class
            array_weight: weight of each class
            array_count: pixel count of each class
            confidence_interval: value representing the confidence interval, default is 1.96, indicating the 5-95% interval
        Returns:
            df_error_adjust: error matrix based on the proportions of area
    """

    landcover_types = len(data)
    list_landcover = list(np.arange(1, 1 + landcover_types))  # list to indicate the land cover types

    df_confusion = pd.DataFrame(data=data, columns=list_landcover, index=list_landcover)

    df_err_adjust = pd.DataFrame(df_confusion.iloc[0:landcover_types, 0:landcover_types], index=list_landcover, columns=list_landcover)

    df_err_adjust.loc[:, 'n_count'] = df_err_adjust.sum(axis=1)
    df_err_adjust.loc['n_count', :] = df_err_adjust.sum(axis=0)

    df_err_adjust.loc[:, 'UA'] = np.nan
    df_err_adjust.loc['PA', :] = np.nan

    df_err_adjust.loc[list_landcover, 'weight'] = array_weight  # assign weight

    for i_row in range(0, landcover_types):
        df_err_adjust.iloc[i_row, 0: landcover_types] = df_err_adjust.iloc[i_row, 0: landcover_types] / \
                                                        df_err_adjust.loc[i_row + 1, 'n_count'] * df_err_adjust.loc[i_row + 1, 'weight']
        df_err_adjust.loc[i_row + 1, 'total'] = np.nansum(df_err_adjust.iloc[i_row, 0: landcover_types])

    for i_col in range(0, landcover_types):
        df_err_adjust.loc['total', i_col + 1] = np.nansum(df_err_adjust.iloc[0: landcover_types, i_col])

    # calculate the user's and producer's accuracy
    for i in range(0, landcover_types):
        df_err_adjust.loc['PA', i + 1] = df_err_adjust.iloc[i, i] / df_err_adjust.loc['total', i + 1]
        df_err_adjust.loc[i + 1, 'UA'] = df_err_adjust.iloc[i, i] / df_err_adjust.loc[i + 1, 'total']

    df_err_adjust.loc['total', 'total'] = 1

    # calculate the overall accuracy
    overall_accuracy = np.nansum(np.diag(df_err_adjust.iloc[0: landcover_types, 0: landcover_types].values))
    df_err_adjust.loc['PA', 'UA'] = overall_accuracy

    # calculate the user's accuracy variance
    user_accuracy = df_err_adjust['UA'].values[0: landcover_types]
    variance_user_accuracy = user_accuracy * (1 - user_accuracy) / (df_err_adjust['n_count'].values[0: landcover_types] - 1)
    std_user_accuracy = np.sqrt(variance_user_accuracy)

    # calculate the overall accuracy variance
    variance_overall_accuracy = np.power(array_weight, 2) * user_accuracy * (1 - user_accuracy) / (df_err_adjust['n_count'].values[0: landcover_types] - 1)
    variance_overall_accuracy = np.sum(variance_overall_accuracy)
    std_overall_accuracy = np.sqrt(variance_overall_accuracy)

    # calculate the producer's accuracy variance
    producer_accuracy = df_err_adjust.loc['PA', :][0: landcover_types].values
    variance_producer_accuracy = np.zeros(np.shape(producer_accuracy), dtype=float)

    for j in range(0, landcover_types):

        # calculate the estimated Nj
        N_j_estimated = 0
        for i in range(0, landcover_types):
            N_j_estimated = N_j_estimated + array_count[i] / df_confusion.sum(axis=1).values[i] * df_confusion.iloc[i, j]

        # part 1
        part_1 = np.power(array_count[j] * (1 - producer_accuracy[j]), 2) * user_accuracy[j] * (1 - user_accuracy[j])
        n_j = df_confusion.sum(axis=1).values[j]
        part_1 = part_1 / (n_j - 1)

        # part 2
        part_2 = 0
        for i in range(0, landcover_types):
            if i == j:
                pass
            else:
                n_i = df_confusion.sum(axis=1).values[i]
                n_ij = df_confusion.iloc[i, j]

                tmp_1 = np.power(array_count[i], 2) / n_i * n_ij
                tmp_2 = (1 - n_ij / n_i) / (n_i - 1)

                part_2 = part_2 + tmp_1 * tmp_2

        part_2 = np.power(producer_accuracy[j], 2) * part_2

        # final variance producer accuracy
        variance_producer_accuracy[j] = (part_1 + part_2) / np.power(N_j_estimated, 2)

    # standard deviation of producer's accuracy
    std_producer_accuracy = np.sqrt(variance_producer_accuracy)

    # assign the uncertainty to the output table
    df_err_adjust.loc[:, 'UA_uncertainty'] = np.nan
    df_err_adjust['UA_uncertainty'].values[0: landcover_types] = std_user_accuracy * confidence_interval

    df_err_adjust.loc['PA_uncertainty', :] = np.nan
    df_err_adjust.loc['PA_uncertainty', 1: landcover_types] = std_producer_accuracy * confidence_interval

    df_err_adjust.loc['PA_uncertainty', 'UA_uncertainty'] = std_overall_accuracy * confidence_interval

    # reorder the row and columns
    df_err_adjust = df_err_adjust[list_landcover + ['total', 'UA', 'UA_uncertainty', 'n_count', 'weight']]
    df_err_adjust = df_err_adjust.reindex(list_landcover + ['total', 'PA', 'PA_uncertainty', 'n_count'])

    return df_err_adjust


def get_adjusted_area_and_margin_of_error(data, df_err_adjust, array_count, confidence_interval=1.96):
    """
        Calculate the adjusted area and margin of error based on the error matrix generated by the Good Practice strategy
    Args:
        data: the error matrix of sample counts
        df_err_adjust: dataframe of the error matrix based on the proportions of area
        array_count: pixel count of each class
        confidence_interval:

    Returns:
        array_mapped_area: mapped area in km2
        array_adjusted_area: adjusted area in km2
        stand_error_area: standard error of area estimation at 95% confidence level
    """

    stratum_count = data.shape[0]

    # landcover_types = len(data)
    list_stratum_id = list(np.arange(1, 1 + stratum_count))  # list to indicate the land cover types
    df_confusion = pd.DataFrame(data=data, columns=list_stratum_id, index=list_stratum_id)

    array_weight = df_err_adjust['weight'].values[0: stratum_count]
    array_weight = array_weight.astype(float)

    array_mapped_area = df_err_adjust.loc[:, 'total'].values[0: stratum_count]
    array_adjusted_area = df_err_adjust.loc['total', :].values[0: stratum_count]

    #  get the area in km2
    array_mapped_area = array_mapped_area * 900 / 1000000 * np.nansum(array_count)
    array_adjusted_area = array_adjusted_area * 900 / 1000000 * np.nansum(array_count)

    array_mapped_area = array_mapped_area.astype(float)
    array_adjusted_area = array_adjusted_area.astype(float)

    print(f'mapped area (km2): {np.round(array_mapped_area, 2)}')
    print(f'adjusted area (km2): {np.round(array_adjusted_area, 2)}')
    print(f'map bias: map area - adjusted area {np.round(array_mapped_area - array_adjusted_area, 2)}')

    stand_error_area_proportion = np.zeros(stratum_count, dtype=float)
    for k in range(0, stratum_count):
        sum = 0
        for i in range(0, stratum_count):
            ni = np.nansum(df_confusion.iloc[i, 0:stratum_count].values)
            pik = df_err_adjust.iloc[i, k]

            sum += (array_weight[i] * pik - pik * pik) / (ni - 1)  # Eq.10 in Good Practice paper

        stand_error_area_proportion[k] = np.sqrt(sum)

    print(f'standard_area_proportion {stand_error_area_proportion}')

    stand_error_area = stand_error_area_proportion * np.nansum(array_count) * 900 / 1000000 * confidence_interval
    print(f'standard error of area (km2) estimation at 95% confidence level {np.round(stand_error_area, 2)}')

    return (array_mapped_area, array_adjusted_area, stand_error_area)


def plot_df_confusion(array, stratum_des=None, figsize=(16, 13),
                      title=None, x_label='Reference', y_label='Map'):
    """
        plot the confusion matrix
    """

    if stratum_des is None:
        stratum_des = {'1': 'Deforestation',
                       '2': 'Forest gain',
                       '3': 'Stable forest',
                       '4': 'Stable non forest',
                       }

    df_cm = pd.DataFrame(array, index=stratum_des.values(), columns=stratum_des.values())

    figure, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    cmap = matplotlib.cm.GnBu

    tick_labelsize = 16
    axis_labelsize = 20
    title_size = 24
    annosize = 18
    ticklength = 4
    axes_linewidth = 1.5

    im = sns.heatmap(df_cm, annot=True, annot_kws={"size": annosize}, fmt='d', cmap=cmap, cbar=False)
    im.figure.axes[-1].yaxis.set_tick_params(labelsize=annosize)

    ax.tick_params('y', labelsize=tick_labelsize, direction='out', length=ticklength, width=axes_linewidth, left=True, which='major', rotation=0)
    ax.tick_params('x', labelsize=tick_labelsize, direction='out', length=ticklength, width=axes_linewidth, top=False, which='major', rotation=15)

    ax.set_xlabel(x_label, size=axis_labelsize)
    ax.set_ylabel(y_label, size=axis_labelsize)

    overall_accuracy = np.trace(df_cm.values) / np.sum(df_cm.values)
    ax.set_title(f'{title}:  {np.trace(df_cm.values)}/{df_cm.values.sum()}={np.round(overall_accuracy*100, 1)}', size=title_size)

    plt.tight_layout()
    plt.show()


def calculate_actual_pixel_after_starta_change(array_map_original,
                                               array_map_updated,
                                               array_reference,
                                               array_count_original_stratum,
                                               stratum_count_original=4,
                                               stratum_count_new=4):
    """
        calculate the modified stratum count
        Args:
            array_map_original: the previous mapping results for the old stratum system
            array_map_updated: the updated mapping results for new stratum system
            array_reference: the validation interpreter for the new stratum system
            array_count_original_stratum: the count of each stratum in the previous stratum system
            stratum_count_original: the number of stratum in the previous stratum system
            stratum_count_new: the number of stratum in the new stratum system
    """

    # array to store the pixel count of each stratum in the sample, the size is (4, 4*4)
    # The first four rows indicate the original stratum
    # The columns indicate the pixel count in the pair of (updated map, reference interpretation)
    # For example, in original stratum 1, the pixel count of (updated map, reference interpretation) pairs
    array_modified_stratum_count = np.zeros((stratum_count_original, stratum_count_new * stratum_count_new), dtype=float)

    for i in range(0, stratum_count_original):
        for j in range(0, stratum_count_new):
            for k in range(0, stratum_count_new):
                # get the mask for updated map and reference sample matches in the previous stratum
                mask = (array_map_original == i + 1) & (array_map_updated == j + 1) & (array_reference == k + 1)
                array_modified_stratum_count[i, j * stratum_count_new + k] = np.count_nonzero(mask)

    # convert the count to the proportion
    for i in range(0, stratum_count_original):
        if np.sum(array_modified_stratum_count[i, :]) == 0:
            # if the sum of the original stratum is zero, set the proportion to zero to avoid the NaN value
            array_modified_stratum_count[i, :] = 0
        else:
            array_modified_stratum_count[i, :] = array_modified_stratum_count[i, :] / np.sum(array_modified_stratum_count[i, :])

    # get the actual pixel number in the map
    array_count_actual_pixel_after_strata_change = array_modified_stratum_count.copy()
    for i in range(0, stratum_count_original):
        for j in range(0, stratum_count_new):
            for k in range(0, stratum_count_new):
                array_count_actual_pixel_after_strata_change[i, j * stratum_count_new + k] = array_modified_stratum_count[i, j * stratum_count_new + k] * array_count_original_stratum[i]

    return array_count_actual_pixel_after_strata_change


def get_area_corrected_df_after_strata_change(array_modified_stratum_count_actual_pixel,
                                              strata_change_val,
                                              confidence_interval=1.96,
                                              new_stratum_count=None):
    """
        get the area corrected dataframe based on the modified stratum count
        Args:
            array_modified_stratum_count_actual_pixel: the modified stratum count
            strata_change_val: the object of the class StratVal
            confidence_interval: the confidence interval, default is 1.96, indicating the 5-95% interval
            new_stratum_count: the number of new strata, if None, it will be the same as the original stratum count
                               For example, original stratum count is 4 ([1, 2, 3, 4]), and the new stratum count might be 2 ([1, 2])
    """

    stratum_count = int(np.sqrt(np.shape(array_modified_stratum_count_actual_pixel)[1]))  # Count of new strata

    if new_stratum_count is None:
        new_stratum_count = stratum_count

    total_count = np.sum(array_modified_stratum_count_actual_pixel)

    list_land_cover = list(np.arange(1, 1 + new_stratum_count))  # list to indicate the land cover types

    df_err_adjust = pd.DataFrame(data=np.zeros((new_stratum_count, new_stratum_count), dtype=float),
                                 columns=list_land_cover, index=list_land_cover)
    for i in range(0, new_stratum_count):
        for j in range(0, new_stratum_count):
            updated_cell_value = np.nansum(array_modified_stratum_count_actual_pixel[:, i * stratum_count + j])

            df_err_adjust.iloc[i, j] = updated_cell_value

    df_err_adjust = df_err_adjust / total_count

    for i_row in range(0, new_stratum_count):
        df_err_adjust.loc[i_row + 1, 'total'] = np.nansum(df_err_adjust.iloc[i_row, 0: stratum_count])

    for i_col in range(0, new_stratum_count):
        df_err_adjust.loc['total', i_col + 1] = np.nansum(df_err_adjust.iloc[0: stratum_count, i_col])

    df_err_adjust.loc[:, 'UA'] = np.nan
    df_err_adjust.loc['PA', :] = np.nan

    # calculate the user's and producer's accuracy
    for i in range(0, new_stratum_count):
        df_err_adjust.loc['PA', i + 1] = df_err_adjust.iloc[i, i] / df_err_adjust.loc['total', i + 1]
        df_err_adjust.loc[i + 1, 'UA'] = df_err_adjust.iloc[i, i] / df_err_adjust.loc[i + 1, 'total']

    df_err_adjust.loc['total', 'total'] = np.nansum(df_err_adjust.iloc[0: new_stratum_count, 0: new_stratum_count].values)

    # calculate the overall accuracy
    overall_accuracy = np.nansum(np.diag(df_err_adjust.iloc[0: new_stratum_count, 0: new_stratum_count].values))
    df_err_adjust.loc['PA', 'UA'] = overall_accuracy

    overall_accuracy = strata_change_val.accuracy()
    overall_accuracy_se = strata_change_val.accuracy_se()

    users_accuracy = np.array(list(strata_change_val.users_accuracy().values()))
    producers_accuracy = np.array(list(strata_change_val.producers_accuracy().values()))

    users_accuracy_se = np.array(list(strata_change_val.users_accuracy_se().values()))
    producers_accuracy_se = np.array(list(strata_change_val.producers_accuracy_se().values()))

    # assign the uncertainty to the output table
    df_err_adjust.loc[:, 'UA_uncertainty'] = np.nan
    df_err_adjust['UA_uncertainty'].values[0: new_stratum_count] = users_accuracy_se * confidence_interval

    df_err_adjust.loc['PA_uncertainty', :] = np.nan
    df_err_adjust.loc['PA_uncertainty', 1: new_stratum_count] = producers_accuracy_se * confidence_interval

    df_err_adjust.loc['PA_uncertainty', 'UA_uncertainty'] = overall_accuracy_se * confidence_interval

    # reorder the row and columns
    # df_err_adjust = df_err_adjust[list_landcover + ['total', 'UA', 'UA_uncertainty', ]]
    # df_err_adjust = df_err_adjust.reindex(list_landcover + ['total', 'PA', 'PA_uncertainty'])

    return df_err_adjust


# def main():
if __name__=='__main__':

    # example in "Good Practice" paper
    # array_weight = np.array([0.02, 0.015, 0.320, 0.645], dtype=float)
    # array_count = np.array([200000, 150000, 3200000, 6450000], dtype=float)
    #
    # data = np.array([[66, 0, 5, 4],
    #                  [0, 55, 8, 12],
    #                  [1, 0, 153, 11],
    #                  [2, 1, 9, 313]], dtype=int)
    #
    # df_err_adjust = generate_good_practice_matrix(data, array_weight, array_count, confidence_interval=1.96)
    #
    # get_adjusted_area_and_margin_of_error(data, df_err_adjust, array_count, confidence_interval=1.96)
    #
    # stratum_des = {'1': 'Deforestation',
    #                '2': 'Forest gain',
    #                '3': 'Stable forest',
    #                '4': 'Stable non forest',
    #                }
    #
    # plot_df_confusion(data, stratum_des=stratum_des, figsize=(10, 8),
    #                   title=None,
    #                   x_label='Reference', y_label='Map',)
    #
    # # from the confusion matrix, generate the reference data
    # rows = [
    #     [p + 1, p + 1, q + 1, None]
    #     for p in range(data.shape[0])
    #     for q in range(data.shape[1])
    #     for _ in range(data[p, q])
    # ]
    #
    # df_test = pd.DataFrame(rows, columns=['strata', 'map_class', 'ref_class', 'oa_indicator'])
    #
    # strat_val = StratVal(
    #     strata_list=[1, 2, 3, 4], # List of labels for strata.
    #     class_list=[1, 2, 3, 4], # List of labels for LULC map classes.
    #     n_strata=[200000, 150000, 3200000, 6450000], # List of the total number of pixels in each strata.
    #     samples_df=df_test, # pandas DataFrame of reference data
    #     strata_col="strata", # Column label for strata in `samples_df`
    #     ref_class="ref_class", # Column label for reference classes in `samples_df`
    #     map_class="map_class" # Column label for map classes in `samples_df`
    # )
    #
    # overall_accuracy = strat_val.accuracy()
    # users_accuracy = strat_val.users_accuracy()
    # producers_accuracy = strat_val.producers_accuracy()
    # overall_accuracy_se = strat_val.accuracy_se()
    # users_accuracy_se = strat_val.users_accuracy_se()
    # producers_accuracy_se = strat_val.producers_accuracy_se()
    #
    # print(f"accuracy: {strat_val.accuracy()}")
    # print(f"user's accuracy: {strat_val.users_accuracy()}")
    # print(f"producer's accuracy: {strat_val.producers_accuracy()}")
    # print(f"accuracy se: {strat_val.accuracy_se()}")
    # print(f"user's accuracy se: {strat_val.users_accuracy_se()}")
    # print(f"producers's accuracy se: {strat_val.producers_accuracy_se()}")

    # example in "strata change" paper

    df_sample = pd.DataFrame({'strata': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                              'map_class': [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                             'ref_class': [1, 1, 1, 1, 1, 3, 2, 1, 2, 3, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 2, 2, 1, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2],
                             })

    list_count_original_stata = [40000, 30000, 20000, 10000]

    strata_change_val = StratVal(
        strata_list=[1, 2, 3, 4], # List of labels for strata.
        class_list=[1, 2, 3, 4], # List of labels for LULC map classes.
        n_strata=list_count_original_stata, # array of the total number of pixels in each strata.
        samples_df=df_sample, # pandas DataFrame of reference data
        strata_col="strata", # Column label for strata in `samples_df`
        ref_class="ref_class", # Column label for reference classes in `samples_df`
        map_class="map_class" # Column label for map classes in `samples_df`
    )

    overall_accuracy = strata_change_val.accuracy()
    users_accuracy = strata_change_val.users_accuracy()
    producers_accuracy = strata_change_val.producers_accuracy()
    overall_accuracy_se = strata_change_val.accuracy_se()
    users_accuracy_se = strata_change_val.users_accuracy_se()
    producers_accuracy_se = strata_change_val.producers_accuracy_se()

    # print(f"accuracy: {strata_change_val.accuracy()}")
    # print(f"user's accuracy: {strata_change_val.users_accuracy()}")
    # print(f"producer's accuracy: {strata_change_val.producers_accuracy()}")
    # print(f"accuracy se: {strata_change_val.accuracy_se()}")
    # print(f"user's accuracy se: {strata_change_val.users_accuracy_se()}")
    # print(f"producers's accuracy se: {strata_change_val.producers_accuracy_se()}")

    array_count_actual_pixel_after_strata_change = calculate_actual_pixel_after_starta_change(array_map_original=df_sample['strata'].values,
                                                                                              array_map_updated=df_sample['map_class'].values,
                                                                                              array_reference=df_sample['ref_class'].values,
                                                                                              array_count_original_stratum=list_count_original_stata,
                                                                                              stratum_count_original=len(list_count_original_stata),
                                                                                              stratum_count_new=len(list_count_original_stata))

    df_err_adjust_strata_change = get_area_corrected_df_after_strata_change(array_count_actual_pixel_after_strata_change,
                                                                            strata_change_val,
                                                                            confidence_interval=1)

    # the area adjustment part will be done in the future

