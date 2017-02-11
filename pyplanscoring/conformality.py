#!/usr/bin/env python
# -*- coding: ISO-8859-1 -*-
# Conformality.py
"""dicompyler plugin that calculates congruence between selected structure and an isodose line.
    Python 3.4 port Victor Alves"""

import numpy as np
import numpy.ma as ma
from matplotlib.path import Path


# def calc_conformation_index(rtdose, structure, lowerlimit):
#     """From a selected structure and isodose line, return conformality index."""
#     # Read "A simple scoring ratio to index the conformity of radiosurgical
#     # treatment plans" by Ian Paddick.
#     # J Neurosurg (Suppl 3) 93:219-222, 2000
#
#     sPlanes = structure['planes']
#
#     # Get the dose to pixel LUT
#     doselut = rtdose.GetPatientToPixelLUT()
#
#     # Generate a 2d mesh grid to create a polygon mask in dose coordinates
#     # Code taken from Stack Overflow Answer from Joe Kington:
#     # http://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/3655582
#     # Create vertex coordinates for each grid cell
#     x, y = np.meshgrid(np.array(doselut[0]), np.array(doselut[1]))
#     x, y = x.flatten(), y.flatten()
#     dosegridpoints = np.vstack((x, y)).T
#
#     # Get the dose and image data information
#     dd = rtdose.GetDoseData()
#     id = rtdose.GetImageData()
#
#     PITV = 0  # Rx isodose volume in cc
#     CV = 0  # coverage volume
#
#     # Iterate over each plane in the structure
#     for z, sPlane in sPlanes.items():
#
#         # Get the contours with calculated areas and the largest contour index
#         contours, largestIndex = calculate_contour_areas(sPlane)
#
#         # Get the dose plane for the current structure plane
#         doseplane = rtdose.GetDoseGrid(z) * dd['dosegridscaling'] * 100
#
#         # If there is no dose for the current plane, go to the next plane
#         if not len(doseplane):
#             break
#
#         # Calculate the histogram for each contour
#         for i, contour in enumerate(contours):
#             m = get_contour_mask(doselut, dosegridpoints, contour['data'])
#             PITV_vol, CV_vol = calculate_volume(m, doseplane, lowerlimit,
#                                                 dd, id, structure)
#             PITV = PITV + PITV_vol
#             CV = CV + CV_vol
#
#     # Volume units are given in cm^3
#     PITV /= 1000.0
#     CV /= 1000.0
#
#     return PITV, CV
def CalculateCI(rtdose, structure, lowerlimit):
    """From a selected structure and isodose line, return conformality index."""
    # Read "A simple scoring ratio to index the conformity of radiosurgical
    # treatment plans" by Ian Paddick.
    # J Neurosurg (Suppl 3) 93:219-222, 2000

    sPlanes = structure['planes']

    # Get the dose to pixel LUT
    doselut = rtdose.GetPatientToPixelLUT()

    # Generate a 2d mesh grid to create a polygon mask in dose coordinates
    # Code taken from Stack Overflow Answer from Joe Kington:
    # http://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask/3655582
    # Create vertex coordinates for each grid cell
    x, y = np.meshgrid(np.array(doselut[0]), np.array(doselut[1]))
    x, y = x.flatten(), y.flatten()
    dosegridpoints = np.vstack((x, y)).T

    # Get the dose and image data information
    dd = rtdose.GetDoseData()
    id = rtdose.GetImageData()

    PITV = 0  # Rx isodose volume in cc
    CV = 0  # coverage volume

    # Iterate over each plane in the structure
    for z, sPlane in sPlanes.items():

        # Get the contours with calculated areas and the largest contour index
        contours, largestIndex = calculate_contour_areas(sPlane)

        # Get the dose plane for the current structure plane
        doseplane = rtdose.GetDoseGrid(z) * dd['dosegridscaling'] * 100

        # If there is no dose for the current plane, go to the next plane
        if not len(doseplane):
            break

        # Calculate the histogram for each contour
        for i, contour in enumerate(contours):
            m = get_contour_mask(doselut, dosegridpoints, contour['data'])
            PITV_vol, CV_vol = calculate_volume(m, doseplane, lowerlimit,
                                                dd, id, structure)
            PITV = PITV + PITV_vol
            CV = CV + CV_vol

    # Volume units are given in cm^3
    PITV /= 1000.0
    CV /= 1000.0

    return PITV, CV


def calculate_contour_areas(plane):
    """Calculate the area of each contour for the given plane.
       Additionally calculate and return the largest contour index."""

    # Calculate the area for each contour in the current plane
    contours = []
    largest = 0
    largestIndex = 0
    for c, contour in enumerate(plane):
        # Create arrays for the x,y coordinate pair for the triangulation
        x = []
        y = []
        for point in contour['contourData']:
            x.append(point[0])
            y.append(point[1])

        cArea = 0
        # Calculate the area based on the Surveyor's formula
        for i in range(0, len(x) - 1):
            cArea = cArea + x[i] * y[i + 1] - x[i + 1] * y[i]
        cArea = abs(cArea / 2)
        # Remove the z coordinate from the xyz point tuple
        data = list(map(lambda x: x[0:2], contour['contourData']))
        # Add the contour area and points to the list of contours
        contours.append({'area': cArea, 'data': data})

        # Determine which contour is the largest
        if (cArea > largest):
            largest = cArea
            largestIndex = c

    return contours, largestIndex


def get_contour_mask(doselut, dosegridpoints, contour):
    """Get the mask for the contour with respect to the dose plane."""

    p = Path(contour)
    grid = p.contains_points(dosegridpoints)
    grid = grid.reshape((len(doselut[1]), len(doselut[0])))

    return grid


def calculate_volume(mask, doseplane, lowerlimit, dd, id, structure):
    """Calculate the differential DVH for the given contour and dose plane."""

    # Multiply the structure mask by the dose plane to get the dose mask
    mask = ma.array(doseplane, mask=~mask)

    # Calculate the volume for the contour for the given dose plane
    PITV_vol = np.sum(doseplane > lowerlimit) * ((id['pixelspacing'][0]) *
                                                 (id['pixelspacing'][1]) *
                                                 (structure['thickness']))
    CV_vol = np.sum(mask.compressed() > lowerlimit) * ((id['pixelspacing'][0]) *
                                                       (id['pixelspacing'][1]) *
                                                       (structure['thickness']))
    return PITV_vol, CV_vol
