from __future__ import division

import logging

import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp1d

from pyplanscoring.core.geometry import centroid_of_polygon

logger = logging.getLogger('dicomparser')
import numpy as np

try:
    import pydicom as dicom
except:
    import dicom
import random
from PIL import Image
from math import pow, sqrt


class DicomParser(object):
    """Class that parses and returns formatted DICOM RT data."""
    """Parses DICOM / DICOM RT files."""

    def __init__(self, dataset=None, filename=None):

        if dataset:
            self.ds = dataset
        elif filename:
            try:
                # Only pydicom 0.9.5 and above supports the force read argument
                if dicom.__version__ >= "0.9.5":
                    self.ds = dicom.read_file(filename, defer_size=100, force=True)
                else:
                    self.ds = dicom.read_file(filename, defer_size=100)
            except (EOFError, IOError):
                # Raise the error for the calling method to handle
                raise
            else:
                # Sometimes DICOM files may not have headers, but they should always
                # have a SOPClassUID to declare what type of file it is. If the
                # file doesn't have a SOPClassUID, then it probably isn't DICOM.
                if not "SOPClassUID" in self.ds:
                    raise AttributeError
        else:
            raise AttributeError

            ######################## SOP Class and Instance Methods ########################

    def GetSOPClassUID(self):
        """Determine the SOP Class UID of the current file."""

        if self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.2':
            return 'rtdose'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            return 'rtss'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.5':
            return 'rtplan'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
            return 'ct'
        else:
            return None

    def GetSOPInstanceUID(self):
        """Determine the SOP Class UID of the current file."""

        return self.ds.SOPInstanceUID

    def GetStudyInfo(self):
        """Return the study information of the current file."""

        study = {}
        if 'StudyDescription' in self.ds:
            desc = self.ds.StudyDescription
        else:
            desc = 'No description'
        study['description'] = desc
        # Don't assume that every dataset includes a study UID
        study['id'] = self.ds.SeriesInstanceUID
        if 'StudyInstanceUID' in self.ds:
            study['id'] = self.ds.StudyInstanceUID

        return study

    def GetSeriesInfo(self):
        """Return the series information of the current file."""

        series = {}
        if 'SeriesDescription' in self.ds:
            desc = self.ds.SeriesDescription
        else:
            desc = 'No description'
        series['description'] = desc
        series['id'] = self.ds.SeriesInstanceUID
        # Don't assume that every dataset includes a study UID
        series['study'] = self.ds.SeriesInstanceUID
        if 'StudyInstanceUID' in self.ds:
            series['study'] = self.ds.StudyInstanceUID
        if 'FrameofReferenceUID' in self.ds:
            series['referenceframe'] = self.ds.FrameofReferenceUID

        return series

    def GetReferencedSeries(self):
        """Return the SOP Class UID of the referenced series."""

        if "ReferencedFrameofReferences" in self.ds:
            if "RTReferencedStudies" in self.ds.ReferencedFrameofReferences[0]:
                if "RTReferencedSeries" in self.ds.ReferencedFrameofReferences[0].RTReferencedStudies[0]:
                    if "SeriesInstanceUID" in \
                            self.ds.ReferencedFrameofReferences[0].RTReferencedStudies[0].RTReferencedSeries[0]:
                        return self.ds.ReferencedFrameofReferences[0].RTReferencedStudies[0].RTReferencedSeries[
                            0].SeriesInstanceUID
        else:
            return ''

    def GetFrameofReferenceUID(self):
        """Determine the Frame of Reference UID of the current file."""

        if 'FrameofReferenceUID' in self.ds:
            return self.ds.FrameofReferenceUID
        elif 'ReferencedFrameofReferences' in self.ds:
            return self.ds.ReferencedFrameofReferences[0].FrameofReferenceUID
        else:
            return ''

    def GetReferencedStructureSet(self):
        """Return the SOP Class UID of the referenced structure set."""

        if "ReferencedStructureSets" in self.ds:
            return self.ds.ReferencedStructureSets[0].ReferencedSOPInstanceUID
        else:
            return ''

    def GetReferencedRTPlan(self):
        """Return the SOP Class UID of the referenced RT plan."""

        if "ReferencedRTPlans" in self.ds:
            return self.ds.ReferencedRTPlans[0].ReferencedSOPInstanceUID
        else:
            return ''

    def GetDemographics(self):
        """Return the patient demographics from a DICOM file."""

        # Set up some sensible defaults for demographics
        patient = {'name': 'N/A',
                   'id': 'N/A',
                   'dob': 'None Found',
                   'gender': 'Other'}
        if 'PatientsName' in self.ds:
            name = self.ds.PatientsName
            patient['name'] = name.family_comma_given().replace(',', '').replace('^', ' ').strip()
        if 'PatientID' in self.ds:
            patient['id'] = self.ds.PatientID
        if 'PatientsSex' in self.ds:
            if self.ds.PatientsSex == 'M':
                patient['gender'] = 'Male'
            elif self.ds.PatientsSex == 'F':
                patient['gender'] = 'Female'
        if 'PatientsBirthDate' in self.ds:
            if len(self.ds.PatientsBirthDate):
                patient['dob'] = str(self.ds.PatientsBirthDate)

        return patient

        ################################ Image Methods #################################

    def GetImageData(self):
        """Return the image data from a DICOM file."""

        data = {}

        data['Position'] = self.ds.ImagePositionPatient
        data['orientation'] = self.ds.ImageOrientationPatient
        data['pixelspacing'] = self.ds.PixelSpacing
        data['rows'] = self.ds.Rows
        data['columns'] = self.ds.Columns
        if 'PatientPosition' in self.ds:
            data['patientposition'] = self.ds.PatientPosition

        return data

    def GetImage(self, window=0, level=0):
        """Return the image from a DICOM image storage file."""

        if (window == 0) and (level == 0):
            window, level = self.GetDefaultImageWindowLevel()
        # Rescale the slope and intercept of the image if present
        if 'RescaleIntercept' in self.ds and 'RescaleSlope' in self.ds:
            rescaled_image = self.ds.pixel_array * self.ds.RescaleSlope + \
                             self.ds.RescaleIntercept
        else:
            rescaled_image = self.ds.pixel_array
        image = self.GetLUTValue(rescaled_image, window, level)

        return Image.fromarray(image).convert('L')

    def GetDefaultImageWindowLevel(self):
        """Determine the default window/level for the DICOM image."""

        window, level = 0, 0
        if ('WindowWidth' in self.ds) and ('WindowCenter' in self.ds):
            if isinstance(self.ds.WindowWidth, float):
                window = self.ds.WindowWidth
            elif isinstance(self.ds.WindowWidth, list):
                if len(self.ds.WindowWidth) > 1:
                    window = self.ds.WindowWidth[1]
            if isinstance(self.ds.WindowCenter, float):
                level = self.ds.WindowCenter
            elif isinstance(self.ds.WindowCenter, list):
                if len(self.ds.WindowCenter) > 1:
                    level = self.ds.WindowCenter[1]
        else:
            wmax = 0
            wmin = 0
            # Rescale the slope and intercept of the image if present
            if 'RescaleIntercept' in self.ds and 'RescaleSlope' in self.ds:
                pixel_array = self.ds.pixel_array * self.ds.RescaleSlope + \
                              self.ds.RescaleIntercept
            else:
                pixel_array = self.ds.pixel_array
            if pixel_array.max() > wmax:
                wmax = pixel_array.max()
            if pixel_array.min() < wmin:
                wmin = pixel_array.min()
            # Default window is the range of the data array
            window = int(abs(wmax) + abs(wmin))
            # Default level is the range midpoint minus the window minimum
            level = int(window / 2 - abs(wmin))
        return window, level

    def GetLUTValue(self, data, window, level):
        """Apply the RGB Look-Up Table for the given data and window/level value."""

        lutvalue = np.piecewise(data,
                                [data <= (level - 0.5 - (window - 1) / 2),
                                 data > (level - 0.5 + (window - 1) / 2)],
                                [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])
        # Convert the resultant array to an unsigned 8-bit array to create
        # an 8-bit grayscale LUT since the range is only from 0 to 255
        return np.array(lutvalue, dtype=np.uint8)

    def GetPatientToPixelLUT(self):
        """Get the image transformation matrix from the DICOM standard Part 3
            Section C.7.6.2.1.1"""

        di = self.ds.PixelSpacing[0]
        dj = self.ds.PixelSpacing[1]
        orientation = self.ds.ImageOrientationPatient
        position = self.ds.ImagePositionPatient

        m = np.matrix(
            [[orientation[0] * di, orientation[3] * dj, 0, position[0]],
             [orientation[1] * di, orientation[4] * dj, 0, position[1]],
             [orientation[2] * di, orientation[5] * dj, 0, position[2]],
             [0, 0, 0, 1]])

        x = []
        y = []
        for i in range(0, self.ds.Columns):
            imat = m * np.matrix([[i], [0], [0], [1]])
            x.append(float(imat[0]))
        for j in range(0, self.ds.Rows):
            jmat = m * np.matrix([[0], [j], [0], [1]])
            y.append(float(jmat[1]))

        return x, y

    ########################### RT Structure Set Methods ###########################

    def GetStructureInfo(self):
        """Return the patient demographics from a DICOM file."""

        structure = {}
        structure['label'] = self.ds.StructureSetLabel
        structure['date'] = self.ds.StructureSetDate
        structure['time'] = self.ds.StructureSetTime
        structure['numcontours'] = len(self.ds.ROIContours)

        return structure

    def GetStructures(self):
        """Returns the structures (ROIs) with their coordinates."""

        structures = {}

        # Determine whether this is RT Structure Set file
        if not (self.GetSOPClassUID() == 'rtss'):
            return structures

        # Locate the name and number of each ROI
        if 'StructureSetROIs' in self.ds:
            for item in self.ds.StructureSetROIs:
                data = {}
                number = item.ROINumber
                data['id'] = number
                data['name'] = item.ROIName
                # logger.debug("Found ROI #%s: %s", str(number), data['name'])
                structures[number] = data

        # Determine the type of each structure (PTV, organ, external, etc)
        if 'RTROIObservations' in self.ds:
            for item in self.ds.RTROIObservations:
                number = item.ReferencedROINumber
                structures[number]['RTROIType'] = item.RTROIInterpretedType

        # TODO handle XIO less than 5.00

        # The coordinate data of each ROI is stored within ROIContourSequence
        if 'ROIContours' in self.ds:
            for roi in self.ds.ROIContours:
                number = roi.ReferencedROINumber

                # Generate a random color for the current ROI
                structures[number]['color'] = np.array((
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)), dtype=float)
                # Get the RGB color triplet for the current ROI if it exists
                if 'ROIDisplayColor' in roi:
                    # Make sure the color is not none
                    if not (roi.ROIDisplayColor is None):
                        color = roi.ROIDisplayColor
                    # Otherwise decode values separated by forward slashes
                    else:
                        value = roi[0x3006, 0x002a].repval
                        color = value.strip("'").split("/")
                    # Try to convert the detected value to a color triplet
                    try:
                        structures[number]['color'] = \
                            np.array(color, dtype=float)
                    # Otherwise fail and fallback on the random color
                    except:
                        logger.debug(
                            "Unable to decode display color for ROI #%s",
                            str(number))

                planes = {}
                if 'Contours' in roi:
                    # Locate the contour sequence for each referenced ROI
                    for contour in roi.Contours:
                        # For each plane, initialize a new plane dictionary
                        plane = {}

                        # Determine all the plane properties
                        plane['geometricType'] = contour.ContourGeometricType
                        plane['numContourPoints'] = contour.NumberofContourPoints
                        plane['contourData'] = self.GetContourPoints(contour.ContourData)
                        # Each plane which coincides with a image slice will have a unique ID
                        if 'ContourImages' in contour:
                            plane['UID'] = contour.ContourImages[0].ReferencedSOPInstanceUID

                        # Add each plane to the planes dictionary of the current ROI
                        if 'geometricType' in plane:
                            z = ('%.2f' % plane['contourData'][0][2]).replace('-0', '0')
                            if z not in planes.keys():
                                planes[z] = []
                            planes[z].append(plane)

                # Calculate the plane thickness for the current ROI
                structures[number]['thickness'] = self.CalculatePlaneThickness(planes)

                # Add the planes dictionary to the current ROI
                # print(planes)
                structures[number]['planes'] = planes

        return structures

    def GetContourPoints(self, array):
        """Parses an array of xyz points and returns a array of point dictionaries."""

        return np.asarray(list(zip(*[iter(array)] * 3)))

    def CalculatePlaneThickness(self, planesDict):
        """Calculates the plane thickness for each structure."""

        planes = []
        # Iterate over each plane in the structure
        for z in planesDict.keys():
            planes.append(float(z))
        planes.sort()

        # Determine the thickness
        thickness = 10000
        for n in range(0, len(planes)):
            if n > 0:
                newThickness = planes[n] - planes[n - 1]
                if newThickness < thickness:
                    thickness = newThickness
        # If the thickness was not detected, set it to 0
        if thickness == 10000:
            thickness = 0

        return thickness

    ############################### RT Dose Methods ###############################

    def HasDVHs(self):
        """Returns whether dose-volume histograms (DVHs) exist."""

        if not "DVHs" in self.ds:
            return False
        else:
            return True

    def GetDVHs(self):
        """Returns the dose-volume histograms (DVHs)."""

        self.dvhs = {}

        if self.HasDVHs():
            for item in self.ds.DVHs:
                # Make sure that the DVH has a referenced structure / ROI
                if not 'DVHReferencedROIs' in item:
                    continue
                number = item.DVHReferencedROIs[0].ReferencedROINumber
                # logger.debug("Found DVH for ROI #%s", str(number))
                dvhitem = {}
                # If the DVH is differential, convert it to a cumulative DVH
                if self.ds.DVHs[0].DVHType == 'DIFFERENTIAL':
                    dvhitem['data'] = self.GenerateCDVH(item.DVHData)
                    dvhitem['bins'] = len(dvhitem['data'])
                # Otherwise the DVH is cumulative
                # Remove "filler" values from DVH data array (even values are DVH values)
                else:
                    dvhitem['data'] = np.array(item.DVHData[1::2])
                    dvhitem['bins'] = int(item.DVHNumberofBins)
                dvhitem['type'] = 'CUMULATIVE'
                dvhitem['doseunits'] = item.DoseUnits
                dvhitem['volumeunits'] = item.DVHVolumeUnits
                dvhitem['scaling'] = item.DVHDoseScaling
                if "DVHMinimumDose" in item:
                    dvhitem['min'] = item.DVHMinimumDose
                else:
                    # save the min dose as -1 so we can calculate_integrate it later
                    dvhitem['min'] = -1
                if "DVHMaximumDose" in item:
                    dvhitem['max'] = item.DVHMaximumDose
                else:
                    # save the max dose as -1 so we can calculate_integrate it later
                    dvhitem['max'] = -1
                if "DVHMeanDose" in item:
                    dvhitem['mean'] = item.DVHMeanDose
                else:
                    # save the mean dose as -1 so we can calculate_integrate it later
                    dvhitem['mean'] = -1
                self.dvhs[number] = dvhitem

        return self.dvhs

    def GenerateCDVH(self, data):
        """Generate a cumulative DVH (cDVH) from a differential DVH (dDVH)"""

        dDVH = np.array(data)
        # Separate the dose and volume values into distinct arrays
        dose = data[0::2]
        volume = data[1::2]

        # Get the min and max dose and volume values
        mindose = int(dose[0] * 100)
        maxdose = int(sum(dose) * 100)
        maxvol = sum(volume)

        # Determine the dose values that are missing from the original data
        missingdose = np.ones(mindose) * maxvol

        # Generate the cumulative dose and cumulative volume data
        k = 0
        cumvol = []
        cumdose = []
        while k < len(dose):
            cumvol += [sum(volume[k:])]
            cumdose += [sum(dose[:k])]
            k += 1
        cumvol = np.array(cumvol)
        cumdose = np.array(cumdose) * 100

        # Interpolate the dDVH data for 1 cGy bins
        interpdose = np.arange(mindose, maxdose + 1)
        interpcumvol = np.interp(interpdose, cumdose, cumvol)

        # Append the interpolated values to the missing dose values
        cumDVH = np.append(missingdose, interpcumvol)

        return cumDVH

    def GetDoseGrid(self, z=0, threshold=0.5):
        """
        Return the 2d dose grid for the given slice Position (mm).

        :param z:           Slice Position in mm.
        :param threshold:   Threshold in mm to determine the max difference from z to the closest dose slice without using interpolation.
        :return:            An numpy 2d array of dose points.
        """

        # If this is a multi-frame dose pixel array,
        # determine the offset for each frame
        if 'GridFrameOffsetVector' in self.ds:
            z = float(z)
            # Get the initial dose grid Position (z) in patient coordinates
            imagepatpos = self.ds.ImagePositionPatient[2]
            orientation = self.ds.ImageOrientationPatient[0]
            # Add the Position to the offset vector to determine the
            # z coordinate of each dose plane
            planes = orientation * np.array(self.ds.GridFrameOffsetVector) + \
                     imagepatpos
            frame = -1
            # Check to see if the requested plane exists in the array
            if np.amin(np.fabs(planes - z)) < threshold:
                frame = np.argmin(np.fabs(planes - z))
            # Return the requested dose plane, since it was found
            if not (frame == -1):
                return self.ds.pixel_array[frame]
            # Check whether the requested plane is within the dose grid boundaries
            elif (z < np.amin(planes)) or (z > np.amax(planes)):
                return []
            # The requested plane was not found, so interpolate between planes
            else:
                # Determine the upper and lower bounds
                umin = np.fabs(planes - z)
                ub = np.argmin(umin)
                lmin = umin.copy()
                # Change the minimum value to the max so we can find the 2nd min
                lmin[ub] = np.amax(umin)
                lb = np.argmin(lmin)
                # Fractional distance of dose plane between upper and lower bound
                fz = (z - planes[lb]) / (planes[ub] - planes[lb])
                plane = self.InterpolateDosePlanes(
                    self.ds.pixel_array[ub], self.ds.pixel_array[lb], fz)
                return plane
        else:
            return []

    def InterpolateDosePlanes(self, uplane, lplane, fz):
        """Interpolates a dose plane between two bounding planes at the given relative location."""

        # uplane and lplane are the upper and lower dose plane, between which the new dose plane
        #   will be interpolated.
        # fz is the fractional distance from the bottom to the top, where the new plane is located.
        #   E.g. if fz = 1, the plane is at the upper plane, fz = 0, it is at the lower plane.

        # A simple linear interpolation
        doseplane = fz * uplane + (1.0 - fz) * lplane

        return doseplane

    def GetIsodosePoints(self, z=0, level=100, threshold=0.5):
        """
        Return points for the given isodose level and slice Position from the dose grid.

        :param z:           Slice Position in mm.
        :param threshold:   Threshold in mm to determine the max difference from z to the closest dose slice without using interpolation.
        :param level:       Isodose level in scaled form (multiplied by self.ds.DoseGridScaling)
        :return:            An array of tuples representing isodose points.
        """

        plane = self.GetDoseGrid(z, threshold)
        isodose = (plane >= level).nonzero()
        return zip(isodose[1].tolist(), isodose[0].tolist())

    def InterpolatePlanes(self, ub, lb, location, ubpoints, lbpoints):
        """Interpolates a plane between two bounding planes at the given location."""

        # If the number of points in the upper bound is higher, use it as the starting bound
        # otherwise switch the upper and lower bounds
        if not (len(ubpoints) >= len(lbpoints)):
            lbCopy = lb.copy()
            lb = ub.copy()
            ub = lbCopy.copy()

        plane = []
        # Determine the closest point in the lower bound from each point in the upper bound
        for u, up in enumerate(ubpoints):
            dist = 100000  # Arbitrary large number
            # Determine the distance from each point in the upper bound to each point in the lower bound
            for l, lp in enumerate(lbpoints):
                newDist = sqrt(pow((up[0] - lp[0]), 2) + pow((up[1] - lp[1]), 2) + pow((ub - lb), 2))
                # If the distance is smaller, then linearly interpolate the point
                if newDist < dist:
                    dist = newDist
                    x = lp[0] + (location - lb) * (up[0] - lp[0]) / (ub - lb)
                    y = lp[1] + (location - lb) * (up[1] - lp[1]) / (ub - lb)
            if not (dist == 100000):
                plane.append((int(x), int(y)))
        return plane

    def GetDoseData(self):
        """Return the dose data from a DICOM RT Dose file."""

        data = self.GetImageData()
        data['doseunits'] = self.ds.DoseUnits
        data['dosetype'] = self.ds.DoseType
        data['dosesummationtype'] = self.ds.DoseSummationType
        data['dosegridscaling'] = self.ds.DoseGridScaling
        data['dosemax'] = float(self.ds.pixel_array.max())

        return data

    def GetReferencedBeamNumber(self):
        """Return the referenced beam number (if it exists) from RT Dose."""

        beam = None
        if "ReferencedRTPlans" in self.ds:
            rp = self.ds.ReferencedRTPlans[0]
            if "ReferencedFractionGroups" in rp:
                rf = rp.ReferencedFractionGroups[0]
                if "ReferencedBeams" in rf:
                    if "ReferencedBeamNumber" in rf.ReferencedBeams[0]:
                        beam = rf.ReferencedBeams[0].ReferencedBeamNumber

        return beam

    ############################### RT Plan Methods ###############################

    def GetPlan(self):
        """Returns the plan information."""

        self.plan = {}
        self.plan['label'] = self.ds.RTPlanLabel
        self.plan['date'] = self.ds.RTPlanDate
        self.plan['time'] = self.ds.RTPlanTime
        self.plan['name'] = ''
        self.plan['rxdose'] = 0
        if "DoseReferences" in self.ds:
            for item in self.ds.DoseReferences:
                if item.DoseReferenceStructureType == 'SITE':
                    self.plan['name'] = "N/A"
                    if "DoseReferenceDescription" in item:
                        self.plan['name'] = item.DoseReferenceDescription
                    if 'TargetPrescriptionDose' in item:
                        rxdose = item.TargetPrescriptionDose * 100
                        if rxdose > self.plan['rxdose']:
                            self.plan['rxdose'] = rxdose
                elif item.DoseReferenceStructureType == 'VOLUME':
                    if 'TargetPrescriptionDose' in item:
                        self.plan['rxdose'] = item.TargetPrescriptionDose * 100
        if ("FractionGroups" in self.ds) and (self.plan['rxdose'] == 0):
            fg = self.ds.FractionGroups[0]
            if ("ReferencedBeams" in fg) and ("NumberofFractionsPlanned" in fg):
                beams = fg.ReferencedBeams
                fx = fg.NumberofFractionsPlanned
                for beam in beams:
                    if "BeamDose" in beam:
                        self.plan['rxdose'] += beam.BeamDose * fx * 100

        if "FractionGroups" in self.ds:
            fg = self.ds.FractionGroups[0]
            if "ReferencedBeams" in fg:
                self.plan['fractions'] = fg.NumberofFractionsPlanned
        self.plan['rxdose'] = int(self.plan['rxdose'])
        self.plan['beams'] = self.GetReferencedBeamsInFraction()

        tmp = self.GetStudyInfo()
        self.plan['description'] = tmp['description']
        if 'RTPlanName' in self.ds:
            self.plan['plan_name'] = self.ds.RTPlanName
        else:
            self.plan['plan_name'] = ''
        if 'PatientsName' in self.ds:
            name = self.ds.PatientsName.family_comma_given().replace(',', '').replace('^', ' ').strip()
            self.plan['patient_name'] = name
        else:
            self.plan['patient_name'] = ''
        return self.plan

    def GetReferencedBeamsInFraction(self, fx=0):
        """Return the referenced beams from the specified fraction."""

        beams = {}
        if "Beams" in self.ds:
            bdict = self.ds.Beams
        elif "IonBeams" in self.ds:
            bdict = self.ds.IonBeams
        else:
            return beams

        # Obtain the beam information
        for b in bdict:
            beam = {}
            beam['name'] = b.BeamName if "BeamName" in b else ""
            beam['description'] = b.BeamDescription if "BeamDescription" in b else ""
            beams[b.BeamNumber] = beam

        # Obtain the referenced beam info from the fraction info
        if "FractionGroups" in self.ds:
            fg = self.ds.FractionGroups[fx]
            if "ReferencedBeams" in fg:
                rb = fg.ReferencedBeams
                nfx = fg.NumberofFractionsPlanned
                for b in rb:
                    if "BeamDose" in b:
                        beams[b.ReferencedBeamNumber]['dose'] = b.BeamDose * nfx * 100
                    if 'BeamMeterset' in b:
                        beams[b.ReferencedBeamNumber]['MU'] = b.BeamMeterset
        return beams


class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class Patient(object):
    def __init__(self, plans):
        self.plan = plans
        self.creation_time = plans['creation_time']

    @lazyproperty
    def total_dose(self):
        return self.plan['rxdose']

    @lazyproperty
    def fractions(self):
        return float(self.plan['fractions'])

    @lazyproperty
    def site(self):
        return self.plan['description']

    @lazyproperty
    def fx_dose(self):
        return self.plan['rxdose'] / float(self.plan['fractions'])

    @lazyproperty
    def name(self):
        return self.plan['patient_name'].upper()

    @lazyproperty
    def start_date(self):
        return pd.to_datetime(self.plan['date'])

    @lazyproperty
    def fields(self):
        return pd.DataFrame(self.plan['beams']).ix['MU'].astype(float).round()

    def receive_fraction(self):
        if self.fractions > 0:
            self.fractions -= 1
            return self.fields.sum()

    def __repr__(self):
        txt = self.plan['label'] + '\n' + self.plan['description'] + '\n' + self.plan['plan_name']

        return txt

    def __cmp__(self, other):
        if hasattr(other, 'creation_time'):
            return self.creation_time.__cmp__(other.creation_time)


class ScoringDicomParser(DicomParser):
    def __init__(self, dataset=None, filename=None):
        DicomParser.__init__(self, dataset=dataset, filename=filename)
        self.interp_dose_plane = None

    def get_tps_data(self):
        """
            Get DICOM-RTDOSE file software information
        :return: TPS data
        """
        data = dict()
        data['TPS'] = ''
        data['Manufacturer'] = ''
        data['SoftwareVersions'] = ''
        if 'ManufacturersModelName' in self.ds:
            data['TPS'] = self.ds.ManufacturersModelName
        if 'Manufacturer' in self.ds:
            data['Manufacturer'] = self.ds.Manufacturer

        if 'SoftwareVersions' in self.ds:
            data['SoftwareVersions'] = self.ds.SoftwareVersions

        return data

    def get_iso_position(self):

        iso = self.ds.BeamSequence[0].ControlPointSequence[0].IsocenterPosition

        return np.array(iso, dtype=float)

    def GetPatientToPixelLUT(self):
        """Get the image transformation matrix from the DICOM standard Part 3
            Section C.7.6.2.1.1"""

        di = self.ds.PixelSpacing[0]
        dj = self.ds.PixelSpacing[1]

        orientation = self.ds.ImageOrientationPatient
        position = self.ds.ImagePositionPatient

        m = np.matrix(
            [[orientation[0] * di, orientation[3] * dj, 0, position[0]],
             [orientation[1] * di, orientation[4] * dj, 0, position[1]],
             [orientation[2] * di, orientation[5] * dj, 0, position[2]],
             [0, 0, 0, 1]])

        x = []
        y = []
        for i in range(0, self.ds.Columns):
            imat = m * np.matrix([[i], [0], [0], [1]])
            x.append(float(imat[0]))
        for j in range(0, self.ds.Rows):
            jmat = m * np.matrix([[0], [j], [0], [1]])
            y.append(float(jmat[1]))

        return x, y

    def pixel2lut(self, di, dj, rows, columns):

        orientation = self.ds.ImageOrientationPatient
        position = self.ds.ImagePositionPatient

        m = np.matrix(
            [[orientation[0] * di, orientation[3] * dj, 0, position[0]],
             [orientation[1] * di, orientation[4] * dj, 0, position[1]],
             [orientation[2] * di, orientation[5] * dj, 0, position[2]],
             [0, 0, 0, 1]])

        x = []
        y = []
        for i in range(0, columns):
            imat = m * np.matrix([[i], [0], [0], [1]])
            x.append(float(imat[0]))
        for j in range(0, rows):
            jmat = m * np.matrix([[0], [j], [0], [1]])
            y.append(float(jmat[1]))

        return x, y

    def GetContourPoints(self, array):
        """Parses an array of xyz points and returns a array of point dictionaries."""

        return np.asarray(list(zip(*[iter(array)] * 3)), dtype=float)

    def GetStructures(self):
        """Returns the structures (ROIs) with their coordinates."""

        structures = {}

        # Determine whether this is RT Structure Set file
        if not (self.GetSOPClassUID() == 'rtss'):
            return structures

        # Locate the name and number of each ROI
        if 'StructureSetROISequence' in self.ds:
            for item in self.ds.StructureSetROISequence:
                data = {}
                number = item.ROINumber
                data['id'] = number
                data['name'] = item.ROIName
                # logger.debug("Found ROI #%s: %s", str(number), data['name'])
                structures[number] = data

        # Determine the type of each structure (PTV, organ, external, etc)
        if 'RTROIObservationsSequence' in self.ds:
            for item in self.ds.RTROIObservationsSequence:
                number = item.ReferencedROINumber
                structures[number]['RTROIType'] = item.RTROIInterpretedType

        # The coordinate data of each ROI is stored within ROIContourSequence
        if 'ROIContourSequence' in self.ds:
            for roi in self.ds.ROIContourSequence:
                number = roi.ReferencedROINumber

                # Generate a random color for the current ROI
                structures[number]['color'] = np.array((
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)), dtype=float)
                # Get the RGB color triplet for the current ROI if it exists
                if 'ROIDisplayColor' in roi:
                    # Make sure the color is not none
                    if not (roi.ROIDisplayColor is None):
                        color = roi.ROIDisplayColor
                    # Otherwise decode values separated by forward slashes
                    else:
                        value = roi[0x3006, 0x002a].repval
                        color = value.strip("'").split("/")
                    # Try to convert the detected value to a color triplet
                    try:
                        structures[number]['color'] = \
                            np.array(color, dtype=float)
                    # Otherwise fail and fallback on the random color
                    except:
                        pass

                        # logger.debug(
                        #     "Unable to decode display color for ROI #%s",
                        #     str(number))

                planes = {}
                if 'ContourSequence' in roi:
                    # Locate the contour sequence for each referenced ROI
                    for contour in roi.ContourSequence:
                        # For each plane, initialize a new plane dictionary
                        plane = {}

                        # Determine all the plane properties
                        plane['geometricType'] = contour.ContourGeometricType
                        plane['numContourPoints'] = contour.NumberOfContourPoints
                        contour_data = self.GetContourPoints(contour.ContourData)
                        plane['contourData'] = contour_data
                        # add info about contour centroid

                        # TODO DEBUG centroid calculation on XiO
                        # if contour.roi not in ['POINT', 'ISOCENTER']:
                        try:
                            plane['centroid'] = centroid_of_polygon(contour_data[:, 0], contour_data[:, 1])
                            # Each plane which coincides with a image slice will have a unique ID
                        except:
                            plane['centroid'] = (contour_data[:, 0].mean(), contour_data[:, 1].mean())
                        # plane['centroid'] = np.median(contour_data[:, 0]), np.median(contour_data[:, 1])

                        if 'ContourImages' in contour:
                            plane['UID'] = contour.ContourImages[0].ReferencedSOPInstanceUID

                        # Add each plane to the planes dictionary of the current ROI
                        if 'geometricType' in plane:
                            # Fixed bug on import z Position on -1.0 < z < 0.0 not using #.replace('-0', '0')
                            z = ('%.2f' % plane['contourData'][0][2])
                            if z not in planes.keys():
                                planes[z] = []
                            planes[z].append(plane)

                # Calculate the plane thickness for the current ROI
                structures[number]['thickness'] = self.CalculatePlaneThickness(planes)

                # Add the planes dictionary to the current ROI
                # print(planes)
                structures[number]['planes'] = planes

        return structures

    def get_grid_3d(self):

        # Get the dose to pixel LUT
        doselut = self.GetPatientToPixelLUT()

        # Get the initial self grid Position (z) in patient coordinates
        imagepatpos = self.ds.ImagePositionPatient[2]
        orientation = self.ds.ImageOrientationPatient[0]

        # Add the Position to the offset vector to determine the

        # z coordinate of each dose plane
        z = orientation * np.array(self.ds.GridFrameOffsetVector) + imagepatpos
        x = np.asarray(doselut[0])
        y = np.asarray(doselut[1])

        return x, y, z

    def DoseRegularGridInterpolator(self):

        x, y, z = self.get_grid_3d()

        values = self.ds.pixel_array * float(self.ds.DoseGridScaling) * 100  # 3D dose matrix in cGy

        # ascending x,y z coordinates
        x_coord = np.arange(len(x))
        y_coord = np.arange(len(y))
        z_coord = np.arange(len(z))

        # mapped coordinates
        fx = interp1d(x, x_coord, fill_value='extrapolate')
        fy = interp1d(y, y_coord, fill_value='extrapolate')
        fz = interp1d(z, z_coord, fill_value='extrapolate')

        dose_interp = RegularGridInterpolator((z_coord, y_coord, x_coord), values, bounds_error=False, fill_value=None)

        return dose_interp, (x, y, z), (fx, fy, fz)

    @property
    def global_max(self):

        dose_matrix = self.ds.pixel_array * float(self.ds.DoseGridScaling) * 100  # 3D dose matrix in cGy

        return np.max(dose_matrix)  # 3D dose matrix in cGy

    def HasDVHs(self):
        """Returns whether dose-volume histograms (DVHs) exist."""

        if not "DVHSequence" in self.ds:
            return False
        else:
            return True

    def GetDVHs(self):
        """Returns the dose-volume histograms (DVHs)."""

        self.dvhs = {}

        if self.HasDVHs():
            cGy = 100
            for item in self.ds.DVHSequence:
                # Make sure that the DVH has a referenced structure / ROI
                if not 'DVHReferencedROISequence' in item:
                    continue
                number = item.DVHReferencedROISequence[0].ReferencedROINumber
                # logger.debug("Found DVH for ROI #%s", str(number))
                dvhitem = {}
                # If the DVH is differential, convert it to a cumulative DVH
                if self.ds.DVHSequence[0].DVHType == 'DIFFERENTIAL':
                    dvhitem['data'] = self.GenerateCDVH(item.DVHData)
                    dvhitem['bins'] = len(dvhitem['data'])
                # Otherwise the DVH is cumulative
                # Remove "filler" values from DVH data array (even values are DVH values)
                else:
                    dvhitem['data'] = np.array(item.DVHData[1::2])
                    dvhitem['bins'] = int(item.DVHNumberOfBins)
                dvhitem['type'] = 'CUMULATIVE'

                if item.DoseUnits == 'GY':
                    # print(item.DoseUnits)
                    dvhitem['doseunits'] = 'cGy'
                else:
                    cGy = 1.0

                dvhitem['volumeunits'] = item.DVHVolumeUnits
                dvhitem['scaling'] = item.DVHDoseScaling
                # Convert all point doses to cGy

                if "DVHMinimumDose" in item:
                    dvhitem['min'] = float(item.DVHMinimumDose) * cGy
                else:
                    # save the min dose as -1 so we can calculate_integrate it later
                    dvhitem['min'] = -1
                if "DVHMaximumDose" in item:
                    dvhitem['max'] = float(item.DVHMaximumDose) * cGy
                else:
                    # save the max dose as -1 so we can calculate_integrate it later
                    dvhitem['max'] = -1
                if "DVHMeanDose" in item:
                    dvhitem['mean'] = float(item.DVHMeanDose) * cGy
                else:
                    # save the mean dose as -1 so we can calculate_integrate it later
                    dvhitem['mean'] = -1
                self.dvhs[number] = dvhitem

        return self.dvhs

    def GetSOPClassUID(self):
        """Determine the SOP Class UID of the current file.
            http://www.dicomlibrary.com/dicom/sop/
        """

        if self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.2':
            return 'rtdose'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            return 'rtss'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.5':
            return 'rtplan'
        # Radiation Therapy Ion Plan Storage
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.8':
            return 'rtplan'
        elif self.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
            return 'ct'
        else:
            return None

    ############################### RT Plan Methods ###############################

    def GetPlan(self):
        """Returns the plan information."""

        self.plan = dict()
        self.plan['label'] = self.ds.RTPlanLabel
        self.plan['date'] = self.ds.RTPlanDate
        self.plan['time'] = self.ds.RTPlanTime
        self.plan['name'] = ''
        self.plan['rxdose'] = 0
        if "DoseReferenceSequence" in self.ds:
            for item in self.ds.DoseReferenceSequence:
                if item.DoseReferenceStructureType == 'SITE':
                    self.plan['name'] = "N/A"
                    if "DoseReferenceDescription" in item:
                        self.plan['name'] = item.DoseReferenceDescription
                    if 'TargetPrescriptionDose' in item:
                        rxdose = item.TargetPrescriptionDose * 100
                        if rxdose > self.plan['rxdose']:
                            self.plan['rxdose'] = rxdose
                elif item.DoseReferenceStructureType == 'VOLUME':
                    if 'TargetPrescriptionDose' in item:
                        self.plan['rxdose'] = item.TargetPrescriptionDose * 100
        if ("FractionGroupSequence" in self.ds) and (self.plan['rxdose'] == 0):
            fg = self.ds.FractionGroupSequence[0]
            if ("ReferencedBeamSequence" in fg) and ("NumberofFractionsPlanned" in fg):
                beams = fg.ReferencedBeamSequence
                fx = fg.NumberofFractionsPlanned
                for beam in beams:
                    if "BeamDose" in beam:
                        self.plan['rxdose'] += beam.BeamDose * fx * 100

        if "FractionGroupSequence" in self.ds:
            fg = self.ds.FractionGroupSequence[0]
            if "ReferencedBeamSequence" in fg:
                self.plan['fractions'] = fg.NumberOfFractionsPlanned
        self.plan['rxdose'] = int(self.plan['rxdose'])
        ref_beams = self.GetReferencedBeamsInFraction()
        self.plan['beams'] = ref_beams

        # try estimate the number of isocenters
        isos = np.array([ref_beams[i]['IsocenterPosition'] for i in ref_beams])
        # round to 2 decimals
        isos = np.round(isos, 2)
        dist = np.sqrt(np.sum((isos - isos[0]) ** 2, axis=1))
        self.plan['n_isocenters'] = len(np.unique(dist))

        # Total number of MU
        total_mu = np.sum([ref_beams[b]['MU'] for b in ref_beams if 'MU' in ref_beams[b]])
        self.plan['Plan_MU'] = total_mu

        tmp = self.GetStudyInfo()
        self.plan['description'] = tmp['description']
        if 'RTPlanName' in self.ds:
            self.plan['plan_name'] = self.ds.RTPlanName
        else:
            self.plan['plan_name'] = ''
        if 'PatientsName' in self.ds:
            name = self.ds.PatientsName.family_comma_given().replace(',', '').replace('^', ' ').strip()
            self.plan['patient_name'] = name
        else:
            self.plan['patient_name'] = ''
        return self.plan

    def GetReferencedBeamsInFraction(self, fx=0):
        """Return the referenced beams from the specified fraction."""

        beams = {}
        if "BeamSequence" in self.ds:
            bdict = self.ds.BeamSequence
        elif "IonBeamSequence" in self.ds:
            bdict = self.ds.IonBeamSequence
        else:
            return beams
        # Obtain the beam information
        for bi in bdict:
            beam = {}
            beam['Manufacturer'] = bi.Manufacturer if "Manufacturer" in bi else ""
            beam['InstitutionName'] = bi.InstitutionName if "InstitutionName" in bi else ""
            beam['TreatmentMachineName'] = bi.TreatmentMachineName if "TreatmentMachineName" in bi else ""
            beam['BeamName'] = bi.BeamName if "BeamName" in bi else ""
            beam['SourcetoSurfaceDistance'] = bi.SourcetoSurfaceDistance if "SourcetoSurfaceDistance" in bi else ""
            beam['BeamDescription '] = bi.BeamDescription if "BeamDescription" in bi else ""
            beam['BeamType'] = bi.BeamType if "BeamType" in bi else ""
            beam['RadiationType'] = bi.RadiationType if "RadiationType" in bi else ""
            beam['ManufacturerModelName'] = bi.ManufacturerModelName if "ManufacturerModelName" in bi else ""
            beam['PrimaryDosimeterUnit'] = bi.PrimaryDosimeterUnit if "PrimaryDosimeterUnit" in bi else ""
            beam['NumberofWedges'] = bi.NumberofWedges if "NumberofWedges" in bi else ""
            beam['NumberofCompensators'] = bi.NumberofCompensators if "NumberofCompensators" in bi else ""
            beam['NumberofBoli'] = bi.NumberofBoli if "NumberofBoli" in bi else ""
            beam['NumberofBlocks'] = bi.NumberofBlocks if "NumberofBlocks" in bi else ""
            ftemp = bi.FinalCumulativeMetersetWeight if "FinalCumulativeMetersetWeight" in bi else ""
            beam['FinalCumulativeMetersetWeight'] = ftemp
            beam['NumberofControlPoints'] = bi.NumberofControlPoints if "NumberofControlPoints" in bi else ""
            # adding mlc info from BeamLimitingDeviceSequence
            beam_limits = bi.BeamLimitingDeviceSequence if "BeamLimitingDeviceSequence" in bi else ""
            beam['BeamLimitingDeviceSequence'] = beam_limits

            # Check control points if exists
            if "ControlPointSequence" in bi:
                beam['ControlPointSequence'] = bi.ControlPointSequence
                # control point 0
                cp0 = bi.ControlPointSequence[0]
                # final control point
                final_cp = bi.ControlPointSequence[-1]

                beam['NominalBeamEnergy'] = cp0.NominalBeamEnergy if "NominalBeamEnergy" in cp0 else ""
                beam['DoseRateSet'] = cp0.DoseRateSet if "DoseRateSet" in cp0 else ""
                beam['IsocenterPosition'] = cp0.IsocenterPosition if "IsocenterPosition" in cp0 else ""
                beam['GantryAngle'] = cp0.GantryAngle if "GantryAngle" in cp0 else ""

                # check VMAT delivery
                if 'GantryRotationDirection' in cp0:
                    if cp0.GantryRotationDirection != 'NONE':

                        # VMAT Delivery
                        beam['GantryRotationDirection'] = cp0.GantryRotationDirection \
                            if "GantryRotationDirection" in cp0 else ""

                        # last control point angle
                        if final_cp.GantryRotationDirection == 'NONE':
                            final_angle = bi.ControlPointSequence[-1].GantryAngle \
                                if "GantryAngle" in cp0 else ""
                            beam['GantryFinalAngle'] = final_angle

                btmp = cp0.BeamLimitingDeviceAngle if "BeamLimitingDeviceAngle" in cp0 else ""
                beam['BeamLimitingDeviceAngle'] = btmp
                beam['TableTopEccentricAngle'] = cp0.TableTopEccentricAngle if "TableTopEccentricAngle" in cp0 else ""

                # check beam limits
                if 'BeamLimitingDevicePositionSequence' in cp0:
                    for bl in cp0.BeamLimitingDevicePositionSequence:
                        beam[bl.RTBeamLimitingDeviceType] = bl.LeafJawPositions

            # Ion control point sequence
            if "IonControlPointSequence" in bi:
                beam['IonControlPointSequence'] = bi.IonControlPointSequence
                cp0 = bi.IonControlPointSequence[0]
                beam['NominalBeamEnergyUnit'] = cp0.NominalBeamEnergyUnit if "NominalBeamEnergyUnit" in cp0 else ""
                beam['NominalBeamEnergy'] = cp0.NominalBeamEnergy if "NominalBeamEnergy" in cp0 else ""
                beam['DoseRateSet'] = cp0.DoseRateSet if "DoseRateSet" in cp0 else ""
                beam['IsocenterPosition'] = cp0.IsocenterPosition if "IsocenterPosition" in cp0 else ""
                beam['GantryAngle'] = cp0.GantryAngle if "GantryAngle" in cp0 else ""
                btmp1 = cp0.BeamLimitingDeviceAngle if "BeamLimitingDeviceAngle" in cp0 else ""
                beam['BeamLimitingDeviceAngle'] = btmp1

            # add each beam to beams dict
            beams[bi.BeamNumber] = beam

        # Obtain the referenced beam info from the fraction info
        if "FractionGroupSequence" in self.ds:
            fg = self.ds.FractionGroupSequence[fx]
            if "ReferencedBeamSequence" in fg:
                rb = fg.ReferencedBeamSequence
                nfx = fg.NumberOfFractionsPlanned
                for bi in rb:
                    if "BeamDose" in bi:
                        # dose in cGy
                        beams[bi.ReferencedBeamNumber]['dose'] = bi.BeamDose * nfx * 100
                    if 'BeamMeterset' in bi:
                        beams[bi.ReferencedBeamNumber]['MU'] = bi.BeamMeterset
        return beams

    def GetDoseData(self):
        """Return the dose data from a DICOM RT Dose file.
            DoseUnits - cGy
        """

        data = self.GetImageData()
        data['doseunits'] = 'cGy'
        data['dosetype'] = self.ds.DoseType
        data['dosesummationtype'] = self.ds.DoseSummationType
        data['dosegridscaling'] = self.ds.DoseGridScaling
        data['dosemax'] = self.global_max

        return data


def test_rtss_eclipse(f):
    ds = dicom.read_file(f, force=True)
    print('SpecificCharacterSet: %s' % ds.SpecificCharacterSet)
    print('Manufacturer: %s' % ds.Manufacturer)
    print('ManufacturersModelName: %s' % ds.ManufacturersModelName)
    print('SoftwareVersions: %s' % ds.SoftwareVersions)
    print('Modality: %s' % ds.Modality)
    print('StructureSetDescription: %s' % ds.StructureSetDescription)
    print('StructureSetROISequence length: %i' % len(ds.StructureSetROISequence))
    print('ROIContourSequence length: %i' % len(ds.ROIContourSequence))

    rois = []
    for roi in ds.ROIContours:
        rois.append(roi)

    print(rois)


if __name__ == '__main__':
    # test RS file reading

    rs_file = r'D:\Dropbox\Plan_Competition_Project\scoring_report\dicom_files\RP.1.2.246.352.71.5.584747638204.952069.20170122155706.dcm'
    plan = ScoringDicomParser(filename=rs_file)
    plan_data = plan.GetPlan()
