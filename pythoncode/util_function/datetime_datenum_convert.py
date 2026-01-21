"""
    tools for python datetime and matlab datenum conversion
"""

import numpy as np
import datetime
from datetime import datetime, timedelta


def datenum_to_datetime_matlabversion(datenum):
    python_datetime = datetime.fromordinal(int(datenum)) + \
                      timedelta(days=float(datenum % 1)) - timedelta(days=366)
    return python_datetime


def datetime_to_datenum_matlabversion(dt):
    mdn = dt + timedelta(days=366)
    frac = (dt - datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac


def datenum_to_datetime(datenum):
    python_datetime = datetime.fromordinal(int(datenum))
    return python_datetime

def datetime_to_datenum(dt):
    python_datenum = dt.toordinal()
    return python_datenum

# datetime_nlcd_2016 = datetime(year=2016,month=7,day=1)
# print(datetime_nlcd_2016)
#
# datenum_nlcd_2016 = datetime_to_datenum(datetime_nlcd_2016)
# print(datenum_nlcd_2016)
#
# datetime_nlcd_2016_converted = datenum_to_datetime(datenum_nlcd_2016)
# print(datetime_nlcd_2016_converted)