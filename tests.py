import unittest
import matplotlib.pyplot as plt
from dicomparser import DicomParser, ScoringDicomParser
import matplotlib.pylab as pl
import numpy as np
import os
# RDOSE
from dosimetric import constrains
from dosimetric import scores
from dvhcalc import get_dvh, get_dvh_pp, calc_dvhs, get_contour_mask, calculate_contour_dvh, calculate_contour_areas, \
    get_cdvh_numba, get_cdvh

from joblib import Parallel, delayed

from scoring import Scoring, get_competition_data, EvalCompetition


def test_getting_data():
    arq = r'D:\Dropbox\Plan_Competition_Project\Eclipse Plans\Venessa IMRT Eclipse\RD-Eclipse-Venessa-IMRTDose.dcm'
    obj_dose = DicomParser(filename=arq)
    dvh = obj_dose.GetDVHs()
    pl.plot(dvh[6]['data'])
    # RSTRUCT
    aqr_file = r'D:\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    obj_str = DicomParser(filename=aqr_file)
    structures = obj_str.GetStructures()
    # RPLAN
    plan_file = r'D:\Dropbox\Plan_Competition_Project\Eclipse Plans\Venessa IMRT Eclipse\RP-Eclipse-Venessa-IMRTPlan.dcm'
    obj_plan = DicomParser(filename=plan_file)
    plan = obj_plan.GetPlan()
    pl.show()


def dvh_estimation_consistency(rs_file, rd_file):
    # Read the example RT structure and RT dose files

    rtss = DicomParser(filename=rs_file)
    rtdose = DicomParser(filename=rd_file)

    # Obtain the structures and DVHs from the DICOM data
    structures = rtss.GetStructures()
    dvhs = rtdose.GetDVHs()

    # Generate the calculated DVHs
    calcdvhs = {}
    for key, structure in structures.items():
        calcdvhs[key] = get_dvh(structure, rtdose)

    # Compare the calculated and original DVH volume for each structure
    print('\nStructure Name\t\t' + 'Original Volume\t\t' + \
          'Calculated Volume\t' + 'Percent Difference')
    print('--------------\t\t' + '---------------\t\t' + \
          '-----------------\t' + '------------------')
    for key, structure in iter(structures.items()):
        if (key in calcdvhs) and (len(calcdvhs[key]['data'])):
            if key in dvhs:
                ovol = dvhs[key]['data'][0]
                cvol = calcdvhs[key]['data'][0]
                print(structure['name'] + '\t' + \
                      str(ovol) + '\t' + \
                      str(cvol) + '\t' + \
                      "%.3f" % float((100) * (cvol - ovol) / (ovol)))

    # Plot the DVHs if pylab is available

    for key, structure in structures.items():
        if (key in calcdvhs) and (len(calcdvhs[key]['data'])):
            if key in dvhs:
                pl.plot(calcdvhs[key]['data'] * 100 / calcdvhs[key]['data'][0],
                        color=np.array(structure['color'], dtype=float) / 255,
                        label=structure['name'], linestyle='dashed')
                pl.plot(dvhs[key]['data'] * 100 / dvhs[key]['data'][0],
                        color=np.array(structure['color'], dtype=float) / 255,
                        label='Original ' + structure['name'])
    pl.legend(loc=7, borderaxespad=-5)
    pl.show()


def test_calc_one_dvh(rs_file, rd_file):
    # rs_file = r'D:\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    # rd_file = r'D:\Dropbox\Plan_Competition_Project\Eclipse Plans\Venessa IMRT Eclipse\RD-Eclipse-Venessa-IMRTDose.dcm'

    rtss = DicomParser(filename=rs_file)
    rtdose = DicomParser(filename=rd_file)
    structures = rtss.GetStructures()
    structure = structures[26]
    dv = get_dvh(structure, rtdose)


def compare_serial_pp_dhv_calc():
    import time

    rs_file = r'D:\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    rd_file = r'D:\Dropbox\Plan_Competition_Project\Eclipse Plans\Venessa IMRT Eclipse\RD-Eclipse-Venessa-IMRTDose.dcm'
    rtss = DicomParser(filename=rs_file)
    rtdose = DicomParser(filename=rd_file)

    # Obtain the structures and DVHs from the DICOM data
    structures = rtss.GetStructures()

    # Generate the calculated DVHs
    st = time.time()
    calcdvhs = {}
    for key, structure in structures.items():
        calcdvhs[key] = get_dvh(structure, rtdose)
        calcdvhs[key]['name'] = structure['name']
    ed = time.time()
    print('Elapsed time serial (seconds): ', ed - st)

    st = time.time()
    res = Parallel(n_jobs=-1)(delayed(get_dvh_pp)(structure, rtdose, key) for key, structure in structures.items())
    cdvh = {}
    for k in res:
        key = k['key']
        del k['key']
        cdvh[key] = k
    ed = time.time()
    print('Elapsed time PP (seconds): ', ed - st)


def test_view_structures():
    # Obtain the structures and DVHs from the DICOM data
    rs_file = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    rd_file = r'C:\Users\Victor\Dropbox\Plan_Competition_Project\Eclipse Plans\Venessa IMRT Eclipse\RD-Eclipse-Venessa-IMRTDose.dcm'

    rtss = DicomParser(filename=rs_file)
    rtdose = DicomParser(filename=rd_file)
    structures = rtss.GetStructures()
    dvhs = {}

    # getting DVH data
    names = []
    if not rtdose.HasDVHs():
        # Generate the calculated DVHs
        dvhs = calc_dvhs(rs_file, rd_file)
    else:
        dvhs = rtdose.GetDVHs()
        for key, structure in structures.items():
            print(structure['name'])
            dvhs[key]['name'] = structure['name']
            names.append(structure['name'])

    x, y, z = [], [], []
    for key, structure in structures.items():
        # sPlanes = structures[16]['planes']
        sPlanes = structure['planes']
        # Iterate over each plane in the structure
        # if structure['name'] == 'BODY':
        #     continue
        for sPlane in sPlanes.values():
            for c, contour in enumerate(sPlane):
                # Create arrays for the x,y coordinate pair for the triangulation
                for point in contour['contourData']:
                    x.append(point[0])
                    y.append(point[1])
                    z.append(point[2])

    import mayavi.mlab as mlab

    mlab.plot3d(x, y, z, color=(1, 0, 0), colormap='spectral')

    mlab.show()


def test_calc_score():
    rs_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'

    # Vanessa
    # rd_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Eclipse Plans\Venessa IMRT Eclipse\RD-Eclipse-Venessa-IMRTDose.dcm'
    # rp_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Eclipse Plans\Venessa IMRT Eclipse\RP-Eclipse-Venessa-IMRTPlan.dcm'
    # dvh_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Eclipse Plans\Venessa IMRT Eclipse\RD-Eclipse-Venessa-IMRTDose.dvh'
    # SAAD
    rd_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Eclipse Plans\Saad RapidArc Eclipse\Saad RapidArc Eclipse\RD.Saad-Eclipse-RapidArc.dcm'
    rp_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Eclipse Plans\Saad RapidArc Eclipse\Saad RapidArc Eclipse\RP.Saad-Eclipse-RapidArc.dcm'
    dvh_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Eclipse Plans\Saad RapidArc Eclipse\Saad RapidArc Eclipse\RD.Saad-Eclipse-RapidArc.dvh'
    # rp = DicomParser(filename=rp_file)

    ## EVAL SCORE
    obj = Scoring(rd_file, rs_file, rp_file, constrains, scores)
    obj.set_dvh_data(dvh_file)
    obj.save_score_results('saad_results.xlsx')
    # obj.calc_dvh_data()
    print(obj.total_score)


def batch_call_dvh(root_path, rs_file, clean_files=False):
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()

    data = get_competition_data(root_path)

    if clean_files:
        dvh_files = [os.path.join(root, name) for root, dirs, files in os.walk(root_path) for name in files if
                     name.endswith('.dvh')]
        for dv in dvh_files:
            os.remove(dv)

    mask = data[1] == 'rtdose'
    rd_files = data['index'][mask].values
    names = data[0][mask].values

    rtss = ScoringDicomParser(filename=rs_file)
    structures = rtss.GetStructures()

    i = 0
    for f, n in zip(rd_files, names):
        p = os.path.splitext(f)
        out_file = p[0] + '.dvh'
        dest, df = os.path.split(f)
        if not os.path.exists(out_file):
            print('Iteration: %i' % i)
            print('processing file: %s' % f)
            calcdvhs = calc_dvhs(n, rs_file, f, out_file=out_file)
            i += 1
            print('processing file done %s' % f)

            fig, ax = plt.subplots()
            fig.set_figheight(12)
            fig.set_figwidth(20)

            for key, structure in structures.items():
                sname = structure['name']
                ax.plot(calcdvhs[sname]['data'] / calcdvhs[sname]['data'][0] * 100,
                        label=sname, linewidth=2.0, color=np.array(structure['color'], dtype=float) / 255)
                ax.legend(loc=7, borderaxespad=-5)
                ax.set_ylabel('Vol (%)')
                ax.set_xlabel('Dose (cGy)')
                ax.set_title(n + ':' + df)
                fig_name = os.path.join(dest, n + '_RD_calc_DVH.png')
                fig.savefig(fig_name, format='png', dpi=100)

            plt.close('all')


def test_eval_competition_data():
    # TODO EVAL FILE ERRORS
    root_path = r'I:\Plan_competition_data\Final Reports'
    rs_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'

    obj = EvalCompetition(root_path, rs_file, constrains, scores)
    obj.set_data()
    res = obj.calc_scores()
    data = obj.comp_data
    sc = [i for i in res if isinstance(i, tuple)]
    import pandas as pd

    data_name = data.set_index(0)
    data_name = data_name.groupby(data_name.index).first()
    df = pd.DataFrame(sc).set_index(0)
    plan_iq = data_name.ix[df.index]['plan_iq_scores']

    comp = pd.concat([plan_iq, df], axis=1)
    comp['delta'] = comp[1] - comp['plan_iq_scores']
    comp = comp.rename(columns={1: 'py_score'})
    comp.to_excel('Plan_IQ_versus_Python_BODY_DMAX.xls')


def test_plot_calc_dvh(rs_file, rd_file):
    calcdvhs = calc_dvhs('teste_dvh_calc', rs_file, rd_file)
    for key in calcdvhs.keys():
        pl.plot(calcdvhs[key]['data'] / calcdvhs[key]['data'][0],
                label=key, linestyle='dashed')
    pl.legend(loc=7, borderaxespad=-5)
    pl.show()


def debug_dvh_calculation():
    # TODO CHECK DVH CALCULATION ON DICOM-RT PLANS THAT FAILED.

    # rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/Competition Package/DICOM Sets/RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    rs_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()
    # PTV EVAL
    from matplotlib.path import Path

    structure = structures[26]

    rd_file = r'I:\ntfsck.00000000\AHMED-~1.COM\DOSE.Z0100.1_FINAL3DCRT.dcm'
    # rd_file = r'I:\ntfsck.00000000\ARNAUD~1.COM\DOSE.LTBREASTAG.31_prop.dcm'

    rp_file = r'I:\ntfsck.00000000\AHMED-~1.COM\RTXPLAN.Z0100.1_FINAL3DCRT.dcm'
    # rp_file = r'I:\ntfsck.00000000\ARNAUD~1.COM\DOSE.LTBREASTAG.31_prop.dcm\RTXPLAN.LTBREASTAG.31_prop.dcm'
    # dvh_file = r'I:\ntfsck.00000000\AHMED-~1.COM\DOSE.Z0100.1_FINAL3DCRT.dvh'
    obj = ScoringDicomParser(filename=rd_file)
    # test_plot_calc_dvh(rs_file, rd_file=rd_file)
    rd = r'I:\ntfsck.00000000\ARNAUD~1.COM\DOSE.LTBREASTAG.31_prop.dcm'

    ## EVAL SCORE
    obj_xio_ok = ScoringDicomParser(filename=rd)

    # dose = obj_xio_ok
    dose = obj
    # test_plot_calc_dvh(rs_file, rd_file=rd_file)


    # def calculate_dvh(structure, dose, limit=None, callback=None):
    #     """Calculate the differential DVH for the given structure and dose grid."""
    limit = None
    sPlanes = structure['planes']
    # logger.debug("Calculating DVH of %s %s", structure['id'], structure['name'])

    # Get the dose to pixel LUT
    doselut = dose.GetPatientToPixelLUT()

    # Generate a 2d mesh grid to create a polygon mask in dose coordinates
    # Code taken from Stack Overflow Answer from Joe Kington:
    # http://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/3655582
    # Create vertex coordinates for each grid cell
    x, y = np.meshgrid(np.array(doselut[0]), np.array(doselut[1]))
    x, y = x.flatten(), y.flatten()
    dosegridpoints = np.vstack((x, y)).T

    # Create an empty array of bins to store the histogram in cGy
    # only if the structure has contour data or the dose grid exists
    if (len(sPlanes)) and ("PixelData" in dose.ds):

        # Get the dose and image data information
        dd = dose.GetDoseData()
        id = dose.GetImageData()

        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
        # Remove values above the limit (cGy) if specified
        if not (limit is None):
            if limit < maxdose:
                maxdose = limit
        hist = np.zeros(maxdose)
    else:
        hist = np.array([0])
    volume = 0

    plane = 0
    # Iterate over each plane in the structure
    for z, sPlane in sPlanes.items():
        # Get the contours with calculated areas and the largest contour index
        contours, largestIndex = calculate_contour_areas(sPlane)

        # Get the dose plane for the current structure plane
        doseplane = dose.GetDoseGrid(z)
        # If there is no dose for the current plane, go to the next plane
        if not len(doseplane):
            break

        # Calculate the histogram for each contour
        for i, contour in enumerate(contours):
            m = get_contour_mask(doselut, dosegridpoints, contour['data'])
            h, vol = calculate_contour_dvh(m, doseplane, maxdose,
                                           dd, id, structure)
            # If this is the largest contour, just add to the total histogram
            if (i == largestIndex):
                hist += h
                volume += vol
            # Otherwise, determine whether to add or subtract histogram
            # depending if the contour is within the largest contour or not
            else:
                contour['inside'] = False
                for point in contour['data']:
                    p = Path(np.array(contours[largestIndex]['data']))
                    if p.contains_point(point):
                        contour['inside'] = True
                        # Assume if one point is inside, all will be inside
                        break
                # If the contour is inside, subtract it from the total histogram
                if contour['inside']:
                    hist -= h
                    volume -= vol
                # Otherwise it is outside, so add it to the total histogram
                else:
                    hist += h
                    volume += vol
        plane += 1
        # if not (callback is None):
        #     callback(plane, len(sPlanes))
    # Volume units are given in cm^3
    volume /= 1000
    # Rescale the histogram to reflect the total volume
    hist = hist * volume / sum(hist)
    # Remove the bins above the max dose for the structure
    hist = np.trim_zeros(hist, trim='b')

    tst = get_cdvh(hist)

    chist = get_cdvh_numba(hist)
    plt.plot(chist / chist[0])
    plt.title(structure['name'] + ' volume: %1.1f' % volume)
    plt.show()
    # for k, v in structures.items():
    #     print(k, v['name'])
    # return hist

    p = Path(contour['data'])
    grid = p.contains_points(dosegridpoints)
    grid = grid.reshape((len(doselut[1]), len(doselut[0])))
    plt.figure()
    plt.imshow(grid)
    plt.title('STRUCTURE_CONTOUR')
    plt.figure()
    plt.imshow(doseplane)
    plt.title('DOSEPLANE')
    # dd = obj_xio_ok.GetDoseData()
    # id = obj_xio_ok.GetImageData()
    #
    # dd1 = obj.GetDoseData()
    # id1 = obj.GetImageData()


if __name__ == '__main__':
    pass

    # rs_file = r'/home/victor/Dropbox/Plan_Competition_Project/Competition Package/DICOM Sets/RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    rs_file = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Competition Package\DICOM Sets\RS.1.2.246.352.71.4.584747638204.208628.20160204185543.dcm'
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()
    # PTV EVAL
    from matplotlib.path import Path

    structure = structures[26]

    rd_file = r'I:\PLAN_TESTING_DATA\Jelle -DONE VMAT jelle.scheurleer@inholland.nl\LTBREAST_VMAT5longlos1_Dose.dcm'
    # rd_file = r'I:\ntfsck.00000000\ARNAUD~1.COM\DOSE.LTBREASTAG.31_prop.dcm'

    rp_file = r'I:\ntfsck.00000000\AHMED-~1.COM\RTXPLAN.Z0100.1_FINAL3DCRT.dcm'
    # rp_file = r'I:\ntfsck.00000000\ARNAUD~1.COM\DOSE.LTBREASTAG.31_prop.dcm\RTXPLAN.LTBREASTAG.31_prop.dcm'
    # dvh_file = r'I:\ntfsck.00000000\AHMED-~1.COM\DOSE.Z0100.1_FINAL3DCRT.dvh'
    obj = ScoringDicomParser(filename=rd_file)
    # test_plot_calc_dvh(rs_file, rd_file=rd_file)
    rd = r'I:\ntfsck.00000000\ARNAUD~1.COM\DOSE.LTBREASTAG.31_prop.dcm'

    dose = obj
    # test_plot_calc_dvh(rs_file, rd_file=rd_file)


    # def calculate_dvh(structure, dose, limit=None, callback=None):
    #     """Calculate the differential DVH for the given structure and dose grid."""
    limit = None
    sPlanes = structure['planes']
    # logger.debug("Calculating DVH of %s %s", structure['id'], structure['name'])

    # Get the dose to pixel LUT
    doselut = dose.GetPatientToPixelLUT()

    # Generate a 2d mesh grid to create a polygon mask in dose coordinates
    # Code taken from Stack Overflow Answer from Joe Kington:
    # http://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/3655582
    # Create vertex coordinates for each grid cell
    x, y = np.meshgrid(np.array(doselut[0]), np.array(doselut[1]))
    x, y = x.flatten(), y.flatten()
    dosegridpoints = np.vstack((x, y)).T

    # Create an empty array of bins to store the histogram in cGy
    # only if the structure has contour data or the dose grid exists
    if (len(sPlanes)) and ("PixelData" in dose.ds):

        # Get the dose and image data information
        dd = dose.GetDoseData()
        id = dose.GetImageData()

        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
        # Remove values above the limit (cGy) if specified
        if not (limit is None):
            if limit < maxdose:
                maxdose = limit
        hist = np.zeros(maxdose)
    else:
        hist = np.array([0])
    volume = 0

    plane = 0
    # Iterate over each plane in the structure
    for z, sPlane in sPlanes.items():
        # Get the contours with calculated areas and the largest contour index
        contours, largestIndex = calculate_contour_areas(sPlane)

        # Get the dose plane for the current structure plane
        doseplane = dose.GetDoseGrid(z)
        # If there is no dose for the current plane, go to the next plane
        if not len(doseplane):
            break

        # Calculate the histogram for each contour
        for i, contour in enumerate(contours):
            m = get_contour_mask(doselut, dosegridpoints, contour['data'])
            h, vol = calculate_contour_dvh(m, doseplane, maxdose,
                                           dd, id, structure)
            # If this is the largest contour, just add to the total histogram
            if (i == largestIndex):
                hist += h
                volume += vol
            # Otherwise, determine whether to add or subtract histogram
            # depending if the contour is within the largest contour or not
            else:
                contour['inside'] = False
                for point in contour['data']:
                    p = Path(np.array(contours[largestIndex]['data']))
                    if p.contains_point(point):
                        contour['inside'] = True
                        # Assume if one point is inside, all will be inside
                        break
                # If the contour is inside, subtract it from the total histogram
                if contour['inside']:
                    hist -= h
                    volume -= vol
                # Otherwise it is outside, so add it to the total histogram
                else:
                    hist += h
                    volume += vol
        plane += 1
        # if not (callback is None):
        #     callback(plane, len(sPlanes))
    # Volume units are given in cm^3
    volume /= 1000
    # Rescale the histogram to reflect the total volume
    hist = hist * volume / sum(hist)
    # Remove the bins above the max dose for the structure
    hist = np.trim_zeros(hist, trim='b')

    tst = get_cdvh(hist)

    chist = get_cdvh_numba(hist)
    plt.plot(chist / chist[0])
    plt.title(structure['name'] + ' volume: %1.1f' % volume)
    plt.show()
    # for k, v in structures.items():
    #     print(k, v['name'])
    # return hist

