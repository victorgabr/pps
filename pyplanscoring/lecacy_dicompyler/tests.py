from __future__ import division

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from pyplanscoring.core.contours3d import plot_contours
from pyplanscoring.core.dicomparser import DicomParser, ScoringDicomParser
from pyplanscoring.core.dosimetric import constrains
from pyplanscoring.core.dvhcalculation import Structure
from pyplanscoring.core.dvhdoses import get_cdvh_numba
from pyplanscoring.core.scoring import Scoring
from pyplanscoring.lecacy_dicompyler.dvhcalc import get_dvh, get_dvh_pp, calc_dvhs, get_contour_mask, \
    calculate_contour_dvh, \
    calculate_contour_areas, \
    get_cdvh


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
    # read the example RT structure and RT dose files

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
    print('\nStructure Name\t\t' + 'Original volume\t\t' + \
          'Calculated volume\t' + 'Percent Difference')
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

    df = pd.read_excel('/home/victor/Dropbox/Plan_Competition_Project/testdata/analytical_data.xlsx')

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


def test_bounding():
    # /  public-domain code by Darel Rex Finley, 2007

    from pyplanscoring.core.geometry import get_axis_grid, wrap_xy_coordinates
    import pandas as pd

    # TODO test bounding rectangles implementation
    def get_bounding_lut(xmin, xmax, ymin, ymax, delta_mm, grid_delta):
        if delta_mm[0] != 0 and delta_mm[1] != 0:
            xmin -= delta_mm[0]
            xmax += delta_mm[0]
            ymin -= delta_mm[1]
            ymax += delta_mm[1]
            x_lut, x_delta = get_axis_grid(abs(delta_mm[0]), [xmin, xmax])
            y_lut, y_delta = get_axis_grid(abs(delta_mm[1]), [ymin, ymax])
            roi_lut = [x_lut, y_lut]
            return roi_lut
        else:
            x_lut, x_delta = get_axis_grid(abs(grid_delta[0]), [xmin, xmax])
            y_lut, y_delta = get_axis_grid(abs(grid_delta[1]), [ymin, ymax])
            roi_lut = [x_lut, y_lut]
        return roi_lut

    def test_contour_roi_grid(contour_points, grid_delta, fac=1.0):
        x = contour_points[:, 0]
        y = contour_points[:, 1]

        deltas = [(-grid_delta[0] * fac, -grid_delta[1] * fac), (0, 0), (grid_delta[0] * fac, grid_delta[1] * fac)]

        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

        bound_rectangles = [get_bounding_lut(xmin, xmax, ymin, ymax, delta, grid_delta) for delta in deltas]

        return bound_rectangles

    def gradient_info_boundary(contour, grid_delta, mapped_coord, dose_interp, z_c):
        br = test_contour_roi_grid(contour, grid_delta)
        dose_stats = []
        st = {}
        for b in br:
            xi, yi = wrap_xy_coordinates(b, mapped_coord)
            doseplane = dose_interp((z_c, yi, xi))
            dfi = pd.DataFrame(doseplane.flatten())
            des = dfi.describe().T
            sts = des['max'] - des['min']
            dose_stats.append(sts)

        df = pd.concat(dose_stats, axis=1)
        df.columns = ['internal', 'bounding', 'external']

        return df


if __name__ == '__main__':
    import time

    # rd = r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_2mm_Aligned.dcm'
    # rs = r'/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/STRUCTURES/Sphere_20_0.dcm'
    #
    rd = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad/6F RA 2.0mm DoseGrid Feb10/RD.1.2.246.352.71.7.584747638204.1758320.20170210154830.dcm'
    rs = '/home/victor/Dropbox/Plan_Competition_Project/competition_2017/plans/Ahmad/6F RA 2.0mm DoseGrid Feb10/RS.1.2.246.352.71.4.584747638204.248648.20170209152429.dcm'

    rtss = ScoringDicomParser(filename=rs)
    dicom_dose = ScoringDicomParser(filename=rd)
    structures = rtss.GetStructures()
    structure = structures[27]

    struc = Structure(structure)
    dose_range0, cdvh0 = struc.calculate_dvh(dicom_dose, up_sample=True)
    ddvh = get_ddvh(cdvh0)

    up_sample = True
    bin_size = 1
    print(' ----- DVH Calculation -----')
    print('Structure Name: %s - volume (cc) %1.3f' % (struc.name, struc.volume_cc))
    # 3D DOSE TRI-LINEAR INTERPOLATION
    dose_interp, grid_3d, mapped_coord = dicom_dose.DoseRegularGridInterpolator()
    sPlanes, dose_lut, dosegrid_points, grid_delta = struc._prepare_data(grid_3d, up_sample)
    print('End caping:  ' + str(struc.end_cap))
    print('Grid delta (mm): ', grid_delta)

    # wrap coordinates
    # wrap z axis
    z_c, ordered_keys = wrap_z_coordinates(sPlanes, mapped_coord)

    # Create an empty array of bins to store the histogram in cGy
    # only if the structure has contour data or the dose grid exists
    maxdose = dicom_dose.global_max
    nbins = int(maxdose / bin_size)
    hist = np.zeros(nbins)
    volume = 0
    count = 0
    import pandas as pd

    st = time.time()
    plane_stats = []
    internal = []
    bounding = []
    external = []
    for i in range(len(ordered_keys)):
        z = ordered_keys[i]
        sPlane = sPlanes[z]
        print('calculating slice z: %.1f' % float(z))
        # Get the contours with calculated areas and the largest contour index
        contours, largestIndex = calculate_contour_areas_numba(sPlane)

        # If there is no dose for the current plane, go to the next plane
        # if not len(doseplane):
        #     break

        # Calculate the histogram for each contour
        for j, contour in enumerate(contours):
            if j == largestIndex:
                dfc = gradient_info_boundary(contour['data'], grid_delta, mapped_coord, dose_interp, z_c[i])
                it = dfc['internal']
                it.name = z
                bd = dfc['bounding']
                bd.name = z
                ext = dfc['external']
                ext.name = z
                internal.append(it)
                bounding.append(bd)
                external.append(ext)

    df_internal = pd.concat(internal, axis=1).T
    df_internal.columns = ['Internal range(cGy)']
    df_bounding = pd.concat(bounding, axis=1).T
    df_bounding.columns = ['Bounding range (cGy)']
    df_external = pd.concat(external, axis=1).T
    df_external.columns = ['External range (cGy)']

    fig, ax = plt.subplots()
    df_internal.plot(ax=ax)
    df_bounding.plot(ax=ax)
    df_external.plot(ax=ax)
    ax.set_ylabel('Dmax - Dmin (cGy)')
    ax.set_xlabel('Z - slice Position (mm)')
    ax.set_title(structure['name'])
    # else:
    #     inside = check_contour_inside(contour['data'], contours[largestIndex]['data'])
    #     # If the contour is inside, subtract it from the total histogram
    #     if not inside:
    #         dfc = gradient_info_boundary(contour['data'], grid_delta, mapped_coord, dose_interp, z_c[i])
    #         ctr_stats.append(dfc)
    #         internal.append(dfc['internal'])
    #         bounding.append(dfc['bounding'])
    #         external.append(dfc['external'])





    # # Get the dose plane for the current structure contour at plane
    # contour_dose_grid, ctr_dose_lut = get_contour_roi_grid(contour['data'], grid_delta)
    # x_c, y_c = wrap_xy_coordinates(ctr_dose_lut, mapped_coord)
    # doseplane = dose_interp((z_c[i], y_c, x_c))
    # m = get_contour_mask_wn(ctr_dose_lut, contour_dose_grid, contour['data'])
    # h, vol = calculate_contour_dvh(m, doseplane, nbins, maxdose, grid_delta)
    #
    # # get gradient statistics
    #



    # # If this is the largest contour, just add to the total histogram
    # if j == largestIndex:
    #     hist += h
    #     volume += vol
    # # Otherwise, determine whether to add or subtract histogram
    # # depending if the contour is within the largest contour or not
    # else:
    #     inside = check_contour_inside(contour['data'], contours[largestIndex]['data'])
    #     # If the contour is inside, subtract it from the total histogram
    #     if inside:
    #         hist -= h
    #         volume -= vol
    #     # Otherwise it is outside, so add it to the total histogram
    #     else:
    #         hist += h
    #         volume += vol








    # # Volume units are given in cm^3
    # volume /= 1000
    # # Rescale the histogram to reflect the total volume
    # hist = hist * volume / sum(hist)
    #
    # chist = get_cdvh_numba(hist)
    # dhist = (np.arange(0, nbins) / nbins) * maxdose
    # idx = np.nonzero(chist)  # remove 0 volumes from DVH
    # dose_range, cdvh = dhist[idx], chist[idx]
    # # dose_range, cdvh = dhist, chist
    # end = time.time()
    # print('elapsed contour roi (s)', end - st)
    #
    # plt.plot(dose_range, cdvh)
    # plt.hold(True)
    # plt.plot(dose_range0, cdvh0)
    # plt.show()
