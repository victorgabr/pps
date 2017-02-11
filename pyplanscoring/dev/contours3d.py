# -*- coding: utf-8 -*-
# vispy: gallery 10
# Copyright (c) 2015, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Demonstrates use of visual.Markers to create a point cloud with a
standard turntable camera to fly around with and a centered 3D Axis.
"""

import vispy.scene
from vispy.scene import visuals


def plot_contours(pos, symbol='o'):
    #
    # Make a canvas and add simple view
    #
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(pos, symbol=symbol, face_color=(1, 1, 1, .5), size=5)

    view.add(scatter)

    view.camera = 'arcball'  # or try 'arcball'
    # view.camera = 'turntable'
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()
