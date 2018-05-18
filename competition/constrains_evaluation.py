import configparser
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from core.dicom_reader import ScoringDicomParser
from core.dicom_reader import PyDicomParser
from pyplanscoring.competition.utils import get_dicom_data
from pyplanscoring.core.dosimetric import read_scoring_criteria
from pyplanscoring.core.dvhcalculation import load, calc_dvhs_pp  # , save
from pyplanscoring.core.scoring import get_matched_names
from validation.validation import CurveCompare

logger = logging.getLogger('constrains_evaluation')

logging.basicConfig(filename='constrains_evaluation.log', level=logging.DEBUG)

#
# folder = os.getcwd()
folder = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/competition'

# Get calculation defaults
config = configparser.ConfigParser()
config.read(os.path.join(folder, 'constrains_evaluation.ini'))
calculation_options = dict()
calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
calculation_options['use_tps_dvh'] = config.getboolean('DEFAULT', 'use_tps_dvh')
calculation_options['use_tps_structures'] = config.getboolean('DEFAULT', 'use_tps_structures')
calculation_options['up_sampling'] = config.getboolean('DEFAULT', 'up_sampling')
calculation_options['maximum_upsampled_volume_cc'] = config.getfloat('DEFAULT', 'maximum_upsampled_volume_cc')
calculation_options['voxel_size'] = config.getfloat('DEFAULT', 'voxel_size')
calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
calculation_options['save_dvh_figure'] = config.getboolean('DEFAULT', 'save_dvh_figure')
calculation_options['save_dvh_data'] = config.getboolean('DEFAULT', 'save_dvh_data')
calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']


def save1(obj, filename, protocol=-1):
    """
        Saves  Object into a file using gzip and Pickle
    :param obj: Calibration Object
    :param filename: Filename *.fco
    :param protocol: cPickle protocol
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def load1(filename):
    """
        Loads a Calibration Object into a file using gzip and Pickle
    :param filename: Calibration filemane *.fco
    :return: object
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_participant_folder(root_folder):
    return [i[0] for i in os.walk(root_folder) if i[2]]


def get_calculated_dvh_data(participant_folder):
    return [os.path.join(participant_folder, name) for root, dirs, files in os.walk(participant_folder) for name in
            files if name.strip().endswith('.dvh')]


class CompareDVH:
    def __init__(self, root, rs_file=''):
        self.root = root
        self.batch = get_participant_folder(root)
        self.rs_file = rs_file
        self.folder_data = {}

    def set_folder_data(self):
        # agreggating DVH data
        for participant_folder in self.batch:
            p_dicom = get_dicom_data(participant_folder)
            dvh = get_calculated_dvh_data(participant_folder)
            # filter folders with at least RD-FILE
            if any(p_dicom):
                if 'rtdose' in p_dicom[1].values and 'rtplan' in p_dicom[1].values:
                    # get files
                    rd = p_dicom.reset_index().set_index(1).ix['rtdose']['index']
                    rp = p_dicom.reset_index().set_index(1).ix['rtplan']['index']
                    # rs = p_dicom.reset_index().set_index(1).ix['rtss']['index']
                    # TPS DVH
                    self.folder_data[participant_folder] = [rp, self.rs_file, rd, dvh]

        return self.folder_data

    def get_tps_data(self):

        # test load_DVH
        data_curves = {}
        for k, val in self.folder_data.items():
            print('entering folder %s' % k)
            try:
                rd_dcm = PyDicomParser(filename=val[2])
                _, key = os.path.split(k)

                data_curves[key] = rd_dcm.get_tps_data()
            except:
                logger.debug('Error in file %s' % val[2])
        return data_curves

    def get_and_save(self, name):
        # test load_DVH
        data_curves = {}
        for k, val in self.folder_data.items():
            print('entering folder %s' % k)
            try:
                rd_dcm = PyDicomParser(filename=val[2])
                tps_dvh = rd_dcm.GetDVHs()
                py_dvh = val[3][0]
                if tps_dvh and py_dvh:
                    pyplan_dvh = load(py_dvh)['DVH']
                    try:
                        tps_m, py_m = self.match_dvh_data(tps_dvh, pyplan_dvh)
                        data_curves[k] = [tps_m, py_m]
                        print('success!')
                    except:
                        print('failed matching dvhs')
                        logger.debug('error in folder %s' % k)

            except:
                logger.debug('error in folder %s' % k)

        save1(data_curves, os.path.join(self.root, name))

    @staticmethod
    def match_dvh_data(tps_dvh, dvh_py):

        py_keys = [val['key'] for k, val in dvh_py.items()]
        tps_py = {k: tps_dvh[k] for k in py_keys}

        # Change keys to number roi
        py_keys_dict = {}
        for k, val in dvh_py.items():
            val['name'] = k
            py_keys_dict[val['key']] = val

        return tps_py, py_keys_dict


#
# def mad_based_outlier(points, thresh=3.5):
#     if len(points.shape) == 1:
#         points = points[:, None]
#     median = np.median(points, axis=0)
#     diff = np.sum((points - median) ** 2, axis=-1)
#     diff = np.sqrt(diff)
#     med_abs_deviation = np.median(diff)
#
#     modified_z_score = 0.6745 * diff / med_abs_deviation
#
#     return modified_z_score > thresh

def mad_based_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
    Returns:
    --------
        mask : A numobservations-length boolean array.
    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def save_figures(data, tps_name, dest_path):
    plt.style.use('ggplot')
    import time
    #
    st = time.time()
    obj = load1(data)
    ed = time.time()
    #
    print('elapsed (s)', ed - st)

    # curve compare

    st = time.time()
    total = {}
    for key, v in obj.items():
        tps_py, py_keys_dict = v
        res = {}
        for k, val in tps_py.items():
            adose = range(len(val['data']))
            cmp = CurveCompare(adose, val['data'], py_keys_dict[k]['dose_axis'], py_keys_dict[k]['data'])
            res[k] = cmp.stats_paper
        total[key] = res

    ed = time.time()
    print('elapsed (s)', ed - st)

    teste = [pd.DataFrame(v).T for k, v in total.items()]
    df = pd.concat(teste)

    # removing outlier ( structure mismatch)
    mask = mad_based_outlier(df.values)
    comp_data = df.loc[~mask]

    for k, v in tps_py.items():
        try:
            tmp = comp_data.loc[k]
            fig, ax = plt.subplots()
            # tmp.hist(ax=ax)
            ax.hist(tmp['mean'])
            ax.set_xlabel('Mean volume error (%)')
            ax.set_ylabel('Frequency')
            title = py_keys_dict[k]['name'] + tps_name + 'vs PyPlanScoring - N = ' + str((len(tmp['mean'])))
            fig.suptitle(title)
            # plt.show()
            fig_name = os.path.join(dest_path, py_keys_dict[k]['name'] + '.png')
            fig.savefig(fig_name, format='png', dpi=100)
            plt.close('all')
        except:
            print('Structure: %s mismatch' % py_keys_dict[k]['name'])


def calculate_and_save_matched_dvh(root, pyplanscoring_folder):
    cmp = CompareDVH(root=root)
    cmp.set_folder_data()

    # pyplanscoring data
    rs_file = os.path.join(pyplanscoring_folder, 'RS.1.2.246.352.71.4.584747638204.253443.20170222200317.dcm')
    path = os.path.join(pyplanscoring_folder, 'Scoring Criteria.txt')
    constrains, scores, criteria_df = read_scoring_criteria(path)

    structures = PyDicomParser(filename=rs_file).GetStructures()
    # calculation_options = {}

    criteria_structure_names, names_dcm = get_matched_names(criteria_df.index.unique(), structures)
    structure_names = criteria_structure_names
    matched_data = {}
    for i, val in cmp.folder_data.items():
        rp_file = ''
        try:
            rd_dcm = PyDicomParser(filename=val[2])
            tps_dvh = rd_dcm.GetDVHs()
            if tps_dvh:
                dvh_file = ''
                cdvh = calc_dvhs_pp(i,
                                    rd_dcm,
                                    structures,
                                    structure_names,
                                    out_file='',
                                    calculation_options=calculation_options)

                tps_py, py_keys_dict = cmp.match_dvh_data(tps_dvh, cdvh)
                matched_data[i] = (tps_py, py_keys_dict)
        except:
            print('failed matching dvhs')
            logger.debug('error in folder %s' % i)

    save1(matched_data, os.path.join(root, 'Pyplanscoring_no_endcap_versus_eclipse.cmp'))


if __name__ == '__main__':
    root = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/ECLIPSE'
    dest_path = r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/competition/figures/DVH_DIFF/eclipse_no_end_capp/mean_diff'
    save_figures(os.path.join(root, 'Pyplanscoring_no_endcap_versus_eclipse.cmp'), " Eclipse™ ", dest_path)

    # name = 'Ray_tps_pyplanscoring_dvh_curves.dvh'
    # cmp.get_and_save(name)
    #
    # data = os.path.join(root, name)
    # save_figures(data, ' RayStation® ', root)
    #
    # test_file = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/RayStation/RayStation_IMRT/Björn Andersson - RayStation - IMRT - 27 MARCH FINAL - 83.5/RD1.2.752.243.1.1.20170518132648587.6500.68535_ANDERSSON-BJÖRN.dcm'
    # rd_dcm = ScoringDicomParser(filename=test_file)

    # plot all curves
    #
    # for k, v in obj.items():
    #     tps_py, py_keys_dict = v
    #     for k, val in tps_py.items():
    #         fig, ax = plt.subplots()
    #         ax.plot(val['data'] / val['data'][0], label='TPS')
    #         ax.plot(py_keys_dict[k]['dose_axis'], py_keys_dict[k]['data'] / py_keys_dict[k]['data'][0], label='Py')
    #         ax.legend()
    #         plt.show()
    #     #
    #     plt.close('all')

    # cmp.get_and_save()
    #
    # batch = get_participant_folder(root)
    #
    # rs = ''
    # # agreggating DVH data
    # data = {}
    # for participant_folder in batch:
    #     p_dicom = get_dicom_data(participant_folder)
    #     dvh = get_calculated_dvh_data(participant_folder)
    #     # filter folders with at least RD-FILE
    #     if any(p_dicom):
    #         if 'rtdose' in p_dicom[1].values and 'rtplan' in p_dicom[1].values and 'rtss' in p_dicom[1].values:
    #             # get files
    #             rd = p_dicom.reset_index().set_index(1).ix['rtdose']['index']
    #             rp = p_dicom.reset_index().set_index(1).ix['rtplan']['index']
    #             rs = p_dicom.reset_index().set_index(1).ix['rtss']['index']
    #             # TPS DVH
    #             data[participant_folder] = [rp, rs, rd, dvh]
    #
    # # test load_DVH
    # i = 1
    # data_curves = {}
    # for k, val in data.items():
    #     rs_dcm = ScoringDicomParser(filename=val[1])
    #     rd_dcm = ScoringDicomParser(filename=val[2])
    #     tps_dvh = rd_dcm.GetDVHs()
    #     py_dvh = val[3][0]
    #     if tps_dvh and py_dvh:
    #         print(k)
    #         data_curves[k] = [tps_dvh, load(py_dvh)]
    #         if i == 1:
    #             break
    #
    # dv = load(py_dvh)
    # dvh_py = dv['DVH']
    #
    # py_keys = [val['key'] for k, val in dvh_py.items()]
    # tps_py = {k: tps_dvh[k] for k in py_keys if k in tps_dvh}
    #
    # # Change keys to number roi
    # py_keys_dict = {}
    # for k, val in dvh_py.items():
    #     val['name'] = k
    #     py_keys_dict[val['key']] = val
    #
    # for k, val in tps_py.items():
    #     fig, ax = plt.subplots()
    #     ax.plot(val['data'])
    #     ax.plot(py_keys_dict[k]['dose_axis'], py_keys_dict[k]['data'])
    # plt.show()  # {k: bigdict[k] for k in ('l', 'm', 'n')}

    # #
    # save(data_curves, os.path.join(root, 'ecplipse_tps_pyplanscoring_dvh_curves.dvh'))

    #
    # t = '/media/victor/TOURO Mobile/COMPETITION 2017/final_plans/ECPLIPSE_VMAT/Qiang Zhao - Eclipse - VMAT-90.1/RD.1.2.246.352.71.7.832719707971.48597.20170320123326.dcm'
    #
    # rd_dcm = ScoringDicomParser(filename=t)
