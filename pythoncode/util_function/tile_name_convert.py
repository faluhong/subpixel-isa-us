"""
    utils tools to convert the tile name format
"""

import os
import sys
from osgeo import gdal, ogr, gdal_array, gdalconst
from os.path import join, exists
import numpy as np

pwd = os.getcwd()
rootpath = os.path.abspath(join(os.getcwd(), "../.."))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def convert_8_tile_names_to_6_tile_names(list_evaluation_tile):
    """
    convert the 8-tile names to 6-tile names, i.e., from "h003v002" to "h03v02"

    :param list_evaluation_tile:
    :return:
    """

    if isinstance(list_evaluation_tile, list):
        list_evaluation_tile_match = []
        for tile_name in list_evaluation_tile:
            list_evaluation_tile_match.append(tile_name[0] + tile_name[2:5] + tile_name[6:8])
        return list_evaluation_tile_match

    elif isinstance(list_evaluation_tile, str):
        list_evaluation_tile_match = list_evaluation_tile[0] + list_evaluation_tile[2:5] + list_evaluation_tile[6:8]
        return list_evaluation_tile_match
    else:
        return "The input is neither a list nor a string."

def convert_6_tile_names_to_8_tile_names(list_evaluation_tile):
    """
    convert the 6-tile names to 8-tile names, i.e., from "h03v02" to "h003v002"

    :param list_evaluation_tile:
    :return:
    """

    if isinstance(list_evaluation_tile, list):
        list_evaluation_tile_match = []
        for tile_name in list_evaluation_tile:
            list_evaluation_tile_match.append(tile_name[0] + '0' + tile_name[1:4] + '0' + tile_name[4:7])
        return list_evaluation_tile_match

    elif isinstance(list_evaluation_tile, str):
        list_evaluation_tile_match = list_evaluation_tile[0] + '0' + list_evaluation_tile[1:4] + '0' + list_evaluation_tile[4:7]
        return list_evaluation_tile_match
    else:
        return "The input is neither a list nor a string."


def find_h_v_index_from_tile_name(tile_name):
    """
    find the h and v index from the tile name using regular expression

    Args:
        tile_name:
    Returns:
    """

    # Search for the pattern in the string
    pattern = r"h(\d{1,3})v(\d{1,3})"

    # Search for the pattern in the string
    match = re.search(pattern, tile_name)

    if match:
        h_index = int(match.group(1))  # Extract the h index
        v_index = int(match.group(2))  # Extract the v index
        return h_index, v_index
    else:
        print("Pattern not found in the string.")
        return None, None




