"""
    calculate the pixel-level reduction and recovery rates for ISPs during financial crises

    The ISP changes between 2005 and 2008 should be larger than 0.01% to avoid division by zero issues.
"""

import numpy as np
import os
from os.path import join, exists
import sys
from osgeo import gdal, gdal_array, gdalconst
import geopandas as gpd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def read_resampled_isp_data(year, scale_factor):
    """
    Read the resampled ISP data from a GeoTIFF file.
    """

    output_folder_resampled = join(rootpath, 'results', 'conus_isp', 'resampled_conus_isp', f'resample_{30*scale_factor}m')

    output_filename_prefix = 'conus_isp_post_processing_binary_is_ndvi015_sm'

    output_raster = join(output_folder_resampled, f'{output_filename_prefix}_{year}_{30 * scale_factor}m.tif')

    img_data = gdal_array.LoadFile(output_raster)

    return img_data


def mask_invalid_pixel(conus_reduction_pct,
                       conus_recovery_pct,
                       conus_resilience_pct,
                       conus_isp_change_2005_2008,
                       conus_isp_resampled_2020,
                       ):

    flag_strategy = 'v8'
    print('v8 strategy: only keep the ISP > 3% pixels in 2020 to keep urban core pixels, ensure the change during 2005-2008 is not zero')
    mask_valid = (np.abs(conus_isp_change_2005_2008) != 0) & (conus_isp_resampled_2020 > 3)

    conus_reduction_pct[~mask_valid] = np.nan
    conus_recovery_pct[~mask_valid] = np.nan
    conus_resilience_pct[~mask_valid] = np.nan

    return (conus_reduction_pct, conus_recovery_pct, conus_resilience_pct, flag_strategy)


def calculate_bivariate_reduction_recovery_map(conus_reduction_pct,
                                               conus_recovery_pct,
                                               array_color):
    """
        calculate the bivariate reduction-recovery map

        :param conus_reduction_pct:
        :param conus_recovery_pct:
        :param array_color:
        :return:
    """

    # array_threshold_reduction = np.array([-80, -20])
    # array_threshold_recovery = np.array([20, 80])

    array_threshold_reduction = np.array([-60, -45])
    array_threshold_recovery = np.array([45, 55])

    conus_bivariate = np.zeros_like(conus_reduction_pct)

    for i_color in range(0, np.shape(array_color)[0]):
        for j_color in range(0, np.shape(array_color)[1]):

            if i_color == 0:
                mask_recovery = (conus_recovery_pct <= array_threshold_recovery[i_color])
            elif i_color == np.shape(array_color)[0] - 1:
                mask_recovery = (conus_recovery_pct > array_threshold_recovery[i_color - 1])
            else:
                mask_recovery = ((conus_recovery_pct > array_threshold_recovery[i_color - 1])
                                 & (conus_recovery_pct <= array_threshold_recovery[i_color]))

            if j_color == 0:
                mask_reduction = (conus_reduction_pct <= array_threshold_reduction[j_color])
            elif j_color == np.shape(array_color)[1] - 1:
                mask_reduction = (conus_reduction_pct > array_threshold_reduction[j_color - 1])
            else:
                mask_reduction = ((conus_reduction_pct > array_threshold_reduction[j_color - 1])
                                  & (conus_reduction_pct <= array_threshold_reduction[j_color]))

            mask_region = mask_reduction & mask_recovery

            conus_bivariate[mask_region] = i_color * np.shape(array_color)[1] + j_color + 1

    return (conus_bivariate, array_threshold_reduction, array_threshold_recovery)


def plot_resilience_related_map(img,
                                title=None,
                                vmin=-200,
                                vmax=200,
                                cmap='viridis',
                                figsize=(14, 12),
                                ticks_flag=True,
                                cbar_label=' ',
                                ax_plot=None,
                                flag_cbar=True, ):
    """
        plot the resilience related map, such as reduction, recovery, resilience percentage map

        :param img:
        :param title:
        :param vmin: minimum value, default as 0
        :param vmax: maximum value, default as 100
        :param figsize:
        :param ticks_flag:
        :param cbar_label: label of the colorbar
        :return:
    """

    sns.set_style("white")

    # change the value outside the range to nan
    img = img.astype(float)
    img[img < vmin] = np.nan
    img[img > vmax] = np.nan

    cmap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # plot the image
    title_label_size = 25
    axis_tick_label_size = 18
    axis_label_size = 24
    cbar_tick_label_size = 24
    tick_length = 6
    axes_line_width = 1.5

    if ax_plot is None:
        fig, ax_plot = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    matplotlib.rcParams['axes.linewidth'] = axes_line_width
    for i in ax_plot.spines.values():
        i.set_linewidth(axes_line_width)

    im = ax_plot.imshow(img, cmap=cmap, norm=norm, interpolation='nearest')

    ax_plot.tick_params('x', labelsize=axis_tick_label_size, direction='out',
                        length=tick_length, width=axes_line_width, bottom=ticks_flag, labelbottom=ticks_flag, which='major')
    ax_plot.tick_params('y', labelsize=axis_tick_label_size, direction='out',
                        length=tick_length, width=axes_line_width, left=ticks_flag, labelleft=ticks_flag, which='major')

    if flag_cbar:
        # define the colorbar

        divider = make_axes_locatable(ax_plot)
        cax = divider.append_axes("right", size="6%", pad=0.3)

        cb = plt.colorbar(im, cax=cax, cmap=cmap, norm=norm,)

        cb.ax.tick_params(labelsize=cbar_tick_label_size, length=tick_length)
        cb.ax.set_ylabel(cbar_label, size=axis_label_size)

    ax_plot.set_title(title, fontsize=title_label_size)

    plt.tight_layout()
    plt.show()


def main():
# if __name__ =='__main__':

    # 30, 34, 100, 200, 300, 3000
    scale_factor = 100   # scale factor used during resampling

    # read the resampled ISP data, the unit is %
    conus_isp_resampled_2005 = read_resampled_isp_data(2005, scale_factor)
    conus_isp_resampled_2006 = read_resampled_isp_data(2006, scale_factor)
    conus_isp_resampled_2007 = read_resampled_isp_data(2007, scale_factor)
    conus_isp_resampled_2008 = read_resampled_isp_data(2008, scale_factor)
    conus_isp_resampled_2009 = read_resampled_isp_data(2009, scale_factor)
    conus_isp_resampled_2010 = read_resampled_isp_data(2010, scale_factor)
    conus_isp_resampled_2011 = read_resampled_isp_data(2011, scale_factor)

    conus_isp_resampled_2017 = read_resampled_isp_data(2017, scale_factor)
    conus_isp_resampled_2018 = read_resampled_isp_data(2018, scale_factor)
    conus_isp_resampled_2019 = read_resampled_isp_data(2019, scale_factor)
    conus_isp_resampled_2020 = read_resampled_isp_data(2020, scale_factor)

    conus_isp_change_2005_2008 = (conus_isp_resampled_2008 - conus_isp_resampled_2005) / 3.0
    conus_isp_change_2008_2011 = (conus_isp_resampled_2011 - conus_isp_resampled_2008) / 3.0
    conus_isp_change_2017_2020 = (conus_isp_resampled_2020 - conus_isp_resampled_2017) / 3.0

    ##
    # divide the absolution ISP change during 2005 to 2008
    conus_reduction_pct = (conus_isp_change_2008_2011 - conus_isp_change_2005_2008) / np.abs(conus_isp_change_2005_2008) * 100
    conus_recovery_pct = conus_isp_change_2017_2020 / np.abs(conus_isp_change_2005_2008) * 100

    conus_resilience_pct = conus_recovery_pct - conus_reduction_pct

    # set the inf values (divided by zero) to nan
    conus_reduction_pct[np.isinf(conus_reduction_pct)] = np.nan
    conus_recovery_pct[np.isinf(conus_recovery_pct)] = np.nan
    conus_resilience_pct[np.isinf(conus_resilience_pct)] = np.nan

    # filter the outliers
    (conus_reduction_pct_filter,
     conus_recovery_pct_filter,
     conus_resilience_pct_filter,
     flag_strategy) = mask_invalid_pixel(conus_reduction_pct,
                                         conus_recovery_pct,
                                         conus_resilience_pct,
                                         conus_isp_change_2005_2008,
                                         conus_isp_resampled_2020, )

    ##
    array_color = np.array([['#e9e6f1', '#9ccae1', '#4fadcf', ],
                            ['#e39bcb', '#9080be', '#3e64ad', ],
                            ['#de50a6', '#833598', '#2b1a8a'],
                            ])

    # convert to bivariate map
    (conus_bivariate,
     array_threshold_reduction,
     array_threshold_recovery) = calculate_bivariate_reduction_recovery_map(conus_reduction_pct_filter,
                                                                            conus_recovery_pct_filter,
                                                                            array_color)

    plot_resilience_related_map(img=conus_bivariate,
                                title='CONUS ISP Recovery-Reduction Bivariate Map',
                                vmin=1, vmax=9,
                                cmap=mcolors.ListedColormap(array_color.flatten().tolist()),
                                cbar_label='Bivariate Class',
                                )














