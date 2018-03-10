import os

from api.backend import PyPlanScoringKernel
from core.calculation import PyStructure, DVHCalculationMP
from core.io import IOHandler, get_participant_folder_data


def test_dvh_data(lens, body, brain, ptv70, spinal_cord, dose_3d, tmpdir):
    # calculating DVH
    grid_up = (0.2, 0.2, 0.2)
    structures_dicom = [lens, body, brain, ptv70, spinal_cord]
    structures_py = [PyStructure(s) for s in structures_dicom]
    grids = [grid_up, None, None, None, None]
    calc_mp = DVHCalculationMP(dose_3d, structures_py, grids, verbose=True)
    dvh_data = calc_mp.calculate_dvh_mp()

    obj = IOHandler(dvh_data)
    assert obj.dvh_data

    # saving dvh file
    file_path = os.path.join(tmpdir, "test_dvh.dvh")
    obj = IOHandler(dvh_data)
    obj.to_dvh_file(file_path)

    obj = IOHandler(dvh_data)
    f_dvh_dict = obj.read_dvh_file(file_path)
    assert f_dvh_dict == dvh_data

    file_path = os.path.join(tmpdir, "test_json_dvh.jdvh")
    obj = IOHandler(dvh_data)
    obj.to_json_file(file_path)

    obj = IOHandler(dvh_data)
    j_dvh_dict = obj.read_json_file(file_path)

    assert j_dvh_dict == dvh_data


def test_get_participant_folder_data(dicom_folder, tmpdir):
    files_dcm, flag = get_participant_folder_data(dicom_folder)
    assert flag

    files, flag = get_participant_folder_data(tmpdir)
    assert not flag

    # test adding one file to tmpdir
    import shutil

    shutil.copy(files_dcm['rtdose'], tmpdir)
    filesd, flag = get_participant_folder_data(tmpdir)
    assert len(filesd) == 2
    assert not flag

    shutil.copy(files_dcm['rtdose'], tmpdir)
    shutil.copy(files_dcm['rtplan'], tmpdir)
    shutil.copy(files_dcm['rtss'], tmpdir)
    filesd, flag = get_participant_folder_data(tmpdir)
    assert flag


def test_save_formatted_repport(dicom_folder, ini_file_path):
    # given case files
    rs_dvh = os.path.join(dicom_folder, 'RS.dcm')
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
    # save report data
    p_kernel.save_report_data()
