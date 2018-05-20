"""Functions to calculate_integrate minimum, maximum, and mean dose to ROI from a cDVH."""
# Copyright (c) 2009 Roy Keyes (roy.coding)
# Copyright (c) 2011 Aditya Panchal
# This file is part of dicompyler, relased under a BSD license.
#    See the file license.txt included with this distribution, also
#    available at http://code.google.com/p/dicompyler/
# Start - 20 Nov. 2009
# It is assumed that the bin width of the cDVH is fixed at 1 cGy.
# speed up numba by victor

import numpy as np

from . import njit


@njit
def get_dvh_min(dvh):
    '''Return minimum dose to ROI derived from cDVH.'''

    # ROI volume (always receives at least 0% dose)
    v1 = dvh[0]

    j = 1
    jmax = len(dvh) - 1
    mindose = 0
    while j < jmax:
        if dvh[j] < v1:
            mindose = (2 * j - 1) / 2.0
            break
        else:
            j += 1

    return mindose


@njit
def get_dvh_max(dvh, dd):
    '''Return maximum dose to ROI derived from cDVH.'''

    # Calulate dDVH
    ddvh = get_ddvh(dvh, dd)

    maxdose = 0
    j = len(ddvh) - 1
    while j >= 0:
        if ddvh[j] > 0.0:
            maxdose = j + 1
            break
        else:
            j -= 1

    return maxdose


@njit
def get_dvh_median(dvh):
    '''Return median dose to ROI derived from cDVH.'''

    mediandose = 0
    # Half of ROI volume
    v1 = dvh[0] / 2.

    j = 1
    jmax = len(dvh) - 1
    while j < jmax:
        if dvh[j] < v1:
            mediandose = (2 * j - 1) / 2.0
            break
        else:
            j += 1

    return mediandose


@njit
def get_dvh_mean(dvh):
    '''Return mean dose to ROI derived from cDVH.'''

    # Mean dose = total dose / ROI volume

    # volume of ROI
    v1 = dvh[0]

    # Calculate dDVH
    ddvh = get_ddvh(dvh, 1.0)

    # Calculate total dose
    j = 1
    dose = 0
    for d in ddvh[1:]:
        dose += d * j
        j += 1

    meandose = dose / v1

    return meandose


@njit
def get_ddvh(cdvh, dd):
    """
        Return differential DVH from cumulative
    :param cdvh: Cumulative volume DVH
    :param dd: dose scaling e.g. 0.01 Gy
    :return:
    """
    # dDVH is the negative "slope" of the cDVH

    j = 0
    jmax = len(cdvh) - 1
    ddvh = np.zeros(jmax + 1)
    while j < jmax:
        ddvh[j] = cdvh[j] - cdvh[j + 1]
        j += 1
    ddvh[j] = cdvh[j]

    return ddvh / dd


@njit
def get_cdvh_numba(ddvh):
    """Calculate the cumulative DVH from a differential DVH array."""

    # cDVH(x) is Sum (Integral) of dDVH with x as lower limit
    # cdvh = np.zeros_like()
    jmax = len(ddvh)
    cdvh = np.zeros(jmax)
    for j in range(jmax):
        cdvh[j] = np.sum(ddvh[j:])

    return cdvh


#
# def test_all():
#     doses = np.arange(150, 100004)
#     a = get_ddvh_slow(doses)
#     b = get_ddvh(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#     a = get_dvh_min_slow(doses)
#     b = get_dvh_min(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#     a = get_dvh_max_slow(doses)
#     b = get_dvh_max(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#     a = get_dvh_median_slow(doses)
#     b = get_dvh_median(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#     a = get_dvh_mean_slow(doses)
#     b = get_dvh_mean(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#
# if __name__ == '__main__':
#     doses = np.arange(150, 100000000)
#
#     # timings
#
#     a = get_ddvh_slow(doses)
#     b = get_ddvh(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#     a = get_dvh_min_slow(doses)
#     b = get_dvh_min(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#     a = get_dvh_max_slow(doses)
#     b = get_dvh_max(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#     a = get_dvh_median_slow(doses)
#     b = get_dvh_median(doses)
#     np.testing.assert_array_almost_equal(b, a)
#
#     a = get_dvh_mean_slow(doses)
#     b = get_dvh_mean(doses)
#     np.testing.assert_array_almost_equal(b, a)
