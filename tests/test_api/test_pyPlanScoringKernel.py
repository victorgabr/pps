import os

import pytest

from api.backend import PyPlanScoringKernel

# TODO monkey patching DVH data to cut redundant calculations]

def test_parse_dicom_folder(dicom_folder):
    # given
    p_kernel = PyPlanScoringKernel()

    # when having a folder containing RS/RD/RP dicom
    # then
    p_kernel.parse_dicom_folder(dicom_folder)
    assert p_kernel.dcm_files


def test_parse_empty_dicom_folder(tmpdir):
    # given
    p_kernel = PyPlanScoringKernel()

    # when having a folder containing RS/RD/RP dicom
    # then
    with pytest.raises(FileNotFoundError):
        p_kernel.parse_dicom_folder(tmpdir)


def test_setup_case(dicom_folder):
    # given case files
    rs_dvh = os.path.join(dicom_folder, 'RS.1.2.246.352.205.5039724533480738438.3109367781599983491.dcm')
    file_path = os.path.join(dicom_folder, 'Scoring_criteria_2018.xlsx')
    case_name = 'BiLateralLungSBRTCase'

    # when instantiate
    p_kernel = PyPlanScoringKernel()
    p_kernel.setup_case(rs_dvh, file_path, case_name)
    assert p_kernel.case is not None


def test_setup_dvh_calculation(dicom_folder, ini_file_path):
    # given case files
    rs_dvh = os.path.join(dicom_folder, 'RS.1.2.246.352.205.5039724533480738438.3109367781599983491.dcm')
    file_path = os.path.join(dicom_folder, 'Scoring_criteria_2018.xlsx')
    case_name = 'BiLateralLungSBRTCase'

    # when instantiate
    p_kernel = PyPlanScoringKernel()

    # if not setup case before setup dvh calculator
    p_kernel.setup_dvh_calculation(ini_file_path)

    # then
    assert p_kernel.dvh_calculator is None

    # when setup case
    p_kernel = PyPlanScoringKernel()
    p_kernel.setup_case(rs_dvh, file_path, case_name)

    p_kernel.setup_dvh_calculation(ini_file_path)

    # then
    assert p_kernel.dvh_calculator is not None


def test_setup_planning_item(dicom_folder, ini_file_path):
    # given case files
    rs_dvh = os.path.join(dicom_folder, 'RS.1.2.246.352.205.5039724533480738438.3109367781599983491.dcm')
    file_path = os.path.join(dicom_folder, 'Scoring_criteria_2018.xlsx')
    case_name = 'BiLateralLungSBRTCase'

    # when instantiate
    p_kernel = PyPlanScoringKernel()
    p_kernel.setup_planing_item()
    assert p_kernel.planning_item is None

    p_kernel.parse_dicom_folder(dicom_folder)
    p_kernel.setup_planing_item()

    assert p_kernel.planning_item is None

    p_kernel.setup_case(rs_dvh, file_path, case_name)
    p_kernel.setup_planing_item()

    assert p_kernel.planning_item is None

    p_kernel.setup_dvh_calculation(ini_file_path)
    p_kernel.setup_planing_item()

    assert p_kernel.planning_item is not None


def test_calculate_dvh(dicom_folder, ini_file_path):
    # given case files
    rs_dvh = os.path.join(dicom_folder, 'RS.1.2.246.352.205.5039724533480738438.3109367781599983491.dcm')
    file_path = os.path.join(dicom_folder, 'Scoring_criteria_2018.xlsx')
    case_name = 'BiLateralLungSBRTCase'

    # when instantiate
    p_kernel = PyPlanScoringKernel()
    p_kernel.parse_dicom_folder(dicom_folder)
    p_kernel.setup_case(rs_dvh, file_path, case_name)
    p_kernel.setup_dvh_calculation(ini_file_path)
    p_kernel.setup_planing_item()

    p_kernel.calculate_dvh()

    assert p_kernel.dvh_data


def test_calc_plan_score(dicom_folder, ini_file_path):
    # given case files
    rs_dvh = os.path.join(dicom_folder, 'RS.1.2.246.352.205.5039724533480738438.3109367781599983491.dcm')
    file_path = os.path.join(dicom_folder, 'Scoring_criteria_2018.xlsx')
    case_name = 'BiLateralLungSBRTCase'

    # when instantiate
    p_kernel = PyPlanScoringKernel()
    p_kernel.parse_dicom_folder(dicom_folder)
    p_kernel.setup_case(rs_dvh, file_path, case_name)
    p_kernel.setup_dvh_calculation(ini_file_path)
    p_kernel.setup_planing_item()

    p_kernel.calculate_dvh()
    p_kernel.calc_plan_score()

    assert not p_kernel._report_data_frame.empty
    assert round(p_kernel._total_score) == round(90.01)

    # save report data
    p_kernel.save_report_data()

def test_calc_plan_complexity(test_case,dicom_folder, ini_file_path):
    # given case files
    rs_dvh = os.path.join(dicom_folder, 'RS.1.2.246.352.205.5039724533480738438.3109367781599983491.dcm')
    file_path = os.path.join(dicom_folder, 'Scoring_criteria_2018.xlsx')
    case_name = 'BiLateralLungSBRTCase'

    # when instantiate
    p_kernel = PyPlanScoringKernel()
    p_kernel.parse_dicom_folder(dicom_folder)
    p_kernel.setup_case(rs_dvh, file_path, case_name)
    p_kernel.setup_dvh_calculation(ini_file_path)
    p_kernel.setup_planing_item()

    # calculate plan complexity
    p_kernel.calc_plan_complexity()
    test_case.assertAlmostEqual(p_kernel.plan_complexity, 0.166503597706,places=3)


def test_save_dvh_data(dicom_folder, ini_file_path):
    # given case files
    rs_dvh = os.path.join(dicom_folder, 'RS.1.2.246.352.205.5039724533480738438.3109367781599983491.dcm')
    file_path = os.path.join(dicom_folder, 'Scoring_criteria_2018.xlsx')
    case_name = 'BiLateralLungSBRTCase'

    # when instantiate
    p_kernel = PyPlanScoringKernel()
    p_kernel.parse_dicom_folder(dicom_folder)
    p_kernel.setup_case(rs_dvh, file_path, case_name)
    p_kernel.setup_dvh_calculation(ini_file_path)
    p_kernel.setup_planing_item()
    p_kernel.calculate_dvh()
    p_kernel.save_dvh_data()

