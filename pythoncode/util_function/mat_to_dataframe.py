"""
    convert the mat file to a pandas dataframe
"""

import pandas as pd

def mat_to_dataframe(mat_rec_cg):
    """
    Convert the mat file to a pandas dataframe

    :param mat_rec_cg:
    :return:
    """

    mat_data = [[row for row in line] for line in mat_rec_cg[0]]
    mat_columns = mat_rec_cg.dtype.names
    df_matfile = pd.DataFrame(mat_data, columns=mat_columns)

    return df_matfile