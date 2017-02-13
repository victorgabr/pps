from __future__ import division

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from pyplanscoring.dev.contours3d import plot_contours
from pyplanscoring.dev.dvhcalculation import Structure
from pyplanscoring.dicomparser import DicomParser, ScoringDicomParser
from pyplanscoring.dosimetric import constrains
from pyplanscoring.dvhcalc import get_dvh, get_dvh_pp, calc_dvhs, get_contour_mask, calculate_contour_dvh, \
    calculate_contour_areas, \
    get_cdvh_numba, get_cdvh
from pyplanscoring.scoring import Scoring


# from pyplanscoring.dosimetric import scores


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
    print(obj.get_total_score)


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

    # rd_file = r'I:\ntfsck.00000000\AHMED-~1.COM\DOSE.Z0100.1_FINAL3DCRT.dcm'
    rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RD.PQRT END TO END.Dose_PLAN.dcm'

    # rp_file = r'I:\ntfsck.00000000\AHMED-~1.COM\RTXPLAN.Z0100.1_FINAL3DCRT.dcm'
    # rp_file = r'I:\ntfsck.00000000\ARNAUD~1.COM\DOSE.LTBREASTAG.31_prop.dcm\RTXPLAN.LTBREASTAG.31_prop.dcm'
    # dvh_file = r'I:\ntfsck.00000000\AHMED-~1.COM\DOSE.Z0100.1_FINAL3DCRT.dvh'
    obj = ScoringDicomParser(filename=rd_file)
    # test_plot_calc_dvh(rs_file, rd_file=rd_file)
    # rd = r'I:\ntfsck.00000000\ARNAUD~1.COM\DOSE.LTBREASTAG.31_prop.dcm'

    ## EVAL SCORE
    # obj_xio_ok = ScoringDicomParser(filename=rd)

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


def test_3d_dose_interpolation():
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm'
    # rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm'
    rd_file = r'/home/victor/Dropbox/Plan_Competition_Project/FantomaPQRT/RD.PQRT END TO END.Dose_PLAN.dcm'
    # DVH ORIGINAL
    dose = ScoringDicomParser(filename=rd_file)

    # Get the dose to pixel LUT
    doselut = dose.GetPatientToPixelLUT()
    x = doselut[0]
    y = doselut[1]
    # UPSAMPLING
    xx = np.linspace(doselut[0][0], doselut[0][-1], 1024)
    yy = np.linspace(doselut[1][0], doselut[1][-1], 1024)

    my_interpolating_function, values = dose.DoseRegularGridInterpolator()

    # GENERATE MESH XY TO GET INTERPOLATED PLANE
    xx, yy = np.meshgrid(xx, yy, indexing='xy', sparse=True)
    res = my_interpolating_function((0.8, yy, xx))
    plt.imshow(res)
    plt.title('interpolated')
    plt.figure()
    original = values[41, :, :]
    plt.imshow(original)
    plt.title('original')
    plt.show()


if __name__ == '__main__':
    test_3d_dose_interpolation()
    from matplotlib.path import Path

    rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_30_0.dcm'
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()

    st = 2
    structure = structures[st]
    rd_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_SupInf_3mm_Aligned.dcm'

    obj = ScoringDicomParser(filename=rd_file)

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
        hist = np.zeros(10000)
    else:
        hist = np.array([0])
    volume = 0

    plane = 0
    # Iterate over each plane in the structure
    zval = np.array([z for z, sPlane in sPlanes.items()], dtype=float)
    zval.sort()
    sp = []
    for z, sPlane in sPlanes.items():
        sp += [sPlane, sPlane, sPlane, sPlane, sPlane, sPlane]
    zval = np.linspace(zval.min(), zval.max(), len(sp))

    # for z, sPlane in sPlanes.items():
    for z, sPlane in zip(zval, sp):

        # Get the contours with calculated areas and the largest contour index
        contours, largestIndex = calculate_contour_areas(sPlane)

        # Get the dose plane for the current structure plane
        doseplane = dose.GetDoseGrid(z)
        # plt.figure()
        # plt.imshow(doseplane * dd['dosegridscaling'] * 100)
        # If there is no dose for the current plane, go to the next plane
        if not len(doseplane):
            break

        # Calculate the histogram for each contour
        for i, contour in enumerate(contours):
            m = get_contour_mask(doselut, dosegridpoints, contour['data'])
            h, vol = calculate_contour_dvh(m, doseplane, maxdose,
                                           dd, id, structure)
            # If this is the largest contour, just add to thttp://oglobo.globo.com/he total histogram
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

    import pandas as pd

    df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/dvh_sphere.xlsx')

    adose = df['Dose (cGy)'].values
    advh = df['3.0 mm slice'].values

    plt.plot(chist / chist[0])
    plt.hold(True)
    plt.plot(adose, advh / advh[0])

    # plt.plot(chist)
    plt.title(structure['name'] + ' volume: %1.1f' % volume)

    plt.show()

    # for c, contour in enumerate(sPlane):
    #     # Create arrays for the x,y coordinate pair for the triangulation
    #     x = []
    #     y = []
    #     for point in contour['contourData']:
    #         x.append(point[0])
    #         y.append(point[1])


def test_roi_expansion():
    rs_file = r'/home/victor/Downloads/DVH-Analysis-Data-Etc/STRUCTURES/Spheres/Sphere_30_0.dcm'
    struc = ScoringDicomParser(filename=rs_file)
    structures = struc.GetStructures()
    st = 2
    structure = structures[st]
    struc_teste = Structure(structure)
    planes_expanded = struc_teste.get_expanded_roi(delta_mm=2)
    ex = np.concatenate(planes_expanded)
    pl = np.concatenate(struc_teste.planes)
    data = np.concatenate([ex, pl])
    plot_contours(data)
