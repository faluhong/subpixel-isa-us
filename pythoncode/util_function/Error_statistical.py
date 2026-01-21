"""
    evaluate the statistical error between x and y
"""

import numpy as np
from scipy import stats
from scipy.interpolate import interpn


def Error_statistical(x, y):
    class Output:

        def __init__(self, N, Bias, MAE, RMSE, STD, r_value, R_square, slope, intercept):
            self.N = N
            self.Bias = Bias
            self.MAE = MAE
            self.RMSE = RMSE
            self.STD = STD
            self.r_value = r_value
            self.R_square = R_square
            self.slope = slope
            self.intercept = intercept

        def print_error_stats(self):
            print('N:', self.N)
            print('count of Nan value', np.count_nonzero(~MaskNan))
            print('Bias x-y:', self.Bias)
            print('MAE:', self.MAE)
            print('RMSE:', self.RMSE)
            print('STD:', self.STD)
            print('R_value:', self.r_value)
            print('R_square:', self.R_square)

            print('y =' + str(slope) + '*x+' + str(intercept))
            print()

    bias, MAE, RMSE, STD = np.nanmean(x - y), np.nanmean(np.abs(x - y)), np.sqrt(np.nanmean(np.square(x - y))), np.nanstd(y - x)

    MaskNan = (~np.isnan(x)) & (~np.isnan(y))

    N = len(x[MaskNan])

    try:
        # The correlation might fail if the input x data is constant
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[MaskNan], y[MaskNan])
    except:
        # print('linear regression failed')
        r_value = 0
        slope = 1.0
        intercept = 0.0
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    output_error_statistic = Output(N, bias, MAE, RMSE, STD, r_value, r_value * r_value, slope, intercept)

    return output_error_statistic




