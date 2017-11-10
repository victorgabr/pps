import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from core.dvhdoses import get_cdvh_numba


def get_dose_vol_array(dvh_file, what):
    f = open(dvh_file)
    found = 0
    stage = 0
    lastrow = 0
    dose, vol = [], []
    for l in f:
        if stage and l[:3] == "200":
            lastrow = 1
        if stage:
            l_ = l.split('\t')
            dose.append(np.float(l_[1].strip()))
            vol.append(np.float(l_[2].strip()))
        if lastrow:
            found = 0
            stage = 0
        if found == 0 and str.find(l, what) > 0:
            found = 1
        if found:
            if str.find(l, 'Bin') >= 0: stage = 1
    dose_step = (dose[-1] - dose[0]) / (len(dose) - 1)  # [Gy]
    dose_vol_array = np.zeros((len(dose), 2))

    for dummy in range(len(dose)):
        dose_vol_array[dummy, 0] = dose[dummy]
        dose_vol_array[dummy, 1] = dose_step * vol[dummy] / 100.

    return dose_vol_array


@nb.njit
def calc_Deff(dose, vol, n):
    D_eff = 0.0
    dose_step = (dose[-1] - dose[0]) / (len(dose) - 1)  # [Gy]
    for dummy in range(len(dose)):
        D_eff += dose_step * vol[dummy] * dose[dummy] ** (1 / n)

    D_eff = D_eff ** n / 100.0

    return D_eff


def lkb(Deff, TD50, m, dx):
    # Lyman-Kutcher-Burman Model:
    # solving integral over numerically
    # and return NTCP for all models
    # uses effective dose in LKB model
    t = (Deff - TD50) / (m * TD50)
    num_range = np.arange(-999, t, dx)
    sum_ntcp = 0.0
    for dummy in range(len(num_range)):
        sum_ntcp += np.exp(-1 * num_range[dummy] ** 2 / 2) * dx

    return 1. / np.sqrt(2 * np.pi) * sum_ntcp


def relative_seriality_model(dose_vol_array, s, gamma, TD50):
    # model as described by Galiardi et al. 2000
    prod = 1.
    for dummy in range(len(dose_vol_array[:, 0])):
        Di = dose_vol_array[dummy, 0]
        Vi = dose_vol_array[dummy, 1]
        PDi = 2 ** (-1 * np.exp(np.e * gamma * (1 - Di / TD50)))
        prod = prod * (1 - PDi ** s) ** Vi

    return (1 - prod) ** (1 / s)


if __name__ == '__main__':
    dvh_file = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\pyplanscoring\radiobiology\example_differential_DVH_CC.txt'

    dvh = get_dose_vol_array(dvh_file, 'Tumorbett')
    plt.plot(dvh[:, 0], dvh[:, 1])

    cdvh = get_cdvh_numba(dvh[:, 1])
    plt.plot(dvh[:, 0], cdvh)
    calc_Deff(dvh[:, 0], dvh[:, 1], 1)
