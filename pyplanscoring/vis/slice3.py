"""
    slice3.py - plot 3D data on a uniform tensor-product grid as a set of
    three adjustable xy, yz, and xz plots

    Copyright (c) 2013 Greg von Winckel
    All rights reserved.

    added dicom coordinate system.
    Copyright (c) 2017 Victor Gabriel Leandro Alves, D.Sc.


    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.widgets import Slider


def meshgrid3(x, y, z):
    """ Create a three-dimensional meshgrid """

    nx = len(x)
    ny = len(y)
    nz = len(z)

    xx = np.swapaxes(np.reshape(np.tile(x, (1, ny, nz)), (nz, ny, nx)), 0, 2)
    yy = np.swapaxes(np.reshape(np.tile(y, (nx, 1, nz)), (nx, nz, ny)), 1, 2)
    zz = np.tile(z, (nx, ny, 1))

    return xx, yy, zz


class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps.
       Created by Joe Kington and submitted to StackOverflow on Dec 1 2012
       http://stackoverflow.com/questions/13656387/can-i-make-matplotlib-sliders-more-discrete
    """

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 1)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):

        xy = self.poly.xy
        xy[2] = val, 1
        xy[3] = val, 0
        self.poly.xy = xy

        # Suppress slider label
        self.valtext.set_text('')

        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(val)


class slice3(object):
    def __init__(self, xx, yy, zz, u):
        self.x = xx[:, 0, 0]
        self.y = yy[0, :, 0]
        self.z = zz[0, 0, :]

        self.data = u

        self.fig = plt.figure(1, (20, 7))
        self.ax1 = self.fig.add_subplot(131, aspect='equal')
        self.ax2 = self.fig.add_subplot(132, aspect='equal')
        self.ax3 = self.fig.add_subplot(133, aspect='equal')

        self.xplot_zline = self.ax1.axvline(color='m', linestyle='--', lw=2)
        self.xplot_zline.set_xdata(self.z[0])

        self.xplot_yline = self.ax1.axhline(color='m', linestyle='--', lw=2)
        self.xplot_yline.set_ydata(self.y[0])

        self.yplot_xline = self.ax2.axhline(color='m', linestyle='--', lw=2)
        self.yplot_xline.set_ydata(self.x[0])

        self.yplot_zline = self.ax2.axvline(color='m', linestyle='--', lw=2)
        self.yplot_zline.set_xdata(self.z[0])

        self.zplot_xline = self.ax3.axvline(color='m', linestyle='--', lw=2)
        self.zplot_xline.set_xdata(self.x[0])

        self.zplot_yline = self.ax3.axhline(color='m', linestyle='--', lw=2)
        self.zplot_yline.set_ydata(self.y[0])

        self.xslice = self.ax1.imshow(
            u[0, :, :], extent=(self.z[0], self.z[-1], self.y[0], self.y[-1]))
        self.yslice = self.ax2.imshow(
            u[:, 0, :], extent=(self.z[0], self.z[-1], self.x[0], self.x[-1]))
        self.zslice = self.ax3.imshow(
            u[:, :, 0], extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]))

        # Create and initialize x-slider
        self.sliderax1 = self.fig.add_axes([0.125, 0.08, 0.225, 0.03])
        self.sliderx = DiscreteSlider(
            self.sliderax1, '', 0, len(self.x) - 1, increment=1, valinit=0)
        self.sliderx.on_changed(self.update_x)
        self.sliderx.set_val(0)

        # Create and initialize y-slider
        self.sliderax2 = self.fig.add_axes([0.4, 0.08, 0.225, 0.03])
        self.slidery = DiscreteSlider(
            self.sliderax2, '', 0, len(self.y) - 1, increment=1, valinit=0)
        self.slidery.on_changed(self.update_y)
        self.slidery.set_val(0)

        # Create and initialize z-slider
        self.sliderax3 = self.fig.add_axes([0.675, 0.08, 0.225, 0.03])
        self.sliderz = DiscreteSlider(
            self.sliderax3, '', 0, len(self.z) - 1, increment=1, valinit=0)
        self.sliderz.on_changed(self.update_z)
        self.sliderz.set_val(0)

        z0, z1 = self.ax1.get_xlim()
        x0, x1 = self.ax2.get_ylim()
        y0, y1 = self.ax1.get_ylim()
        self.ax1.set_aspect((z1 - z0) / (y1 - y0))
        self.ax2.set_aspect((z1 - z0) / (x1 - x0))
        self.ax3.set_aspect((x1 - x0) / (y1 - y0))

    def xlabel(self, *args, **kwargs):
        self.ax2.set_ylabel(*args, **kwargs)
        self.ax3.set_xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        self.ax1.set_ylabel(*args, **kwargs)
        self.ax3.set_ylabel(*args, **kwargs)

    def zlabel(self, *args, **kwargs):
        self.ax1.set_xlabel(*args, **kwargs)
        self.ax2.set_xlabel(*args, **kwargs)

    def update_x(self, value):
        self.xslice.set_data(self.data[value, :, :])
        self.yplot_xline.set_ydata(self.x[value])
        self.zplot_xline.set_xdata(self.x[value])

    def update_y(self, value):
        self.yslice.set_data(self.data[:, value, :])
        self.xplot_yline.set_ydata(self.y[value])
        self.zplot_yline.set_ydata(self.y[value])

    def update_z(self, value):
        self.zslice.set_data(self.data[:, :, value])
        self.xplot_zline.set_xdata(self.z[value])
        self.yplot_zline.set_xdata(self.z[value])

    def show(self):
        plt.show()


class DoseSlice3D(object):
    """ Slicer to view DICOM-RT doses

    dose_3D.shape  - (z,y,x)
    z_slice = dose_3D[z_iso, :, :]  # column dose_3D.shape[1] x row dose_3D.shape[2] (x,y)
    y_slice = dose_3D[:, y_iso, :]  # column dose_3D.shape[0] x row dose_3D.shape[2] (x,z)
    x_slice = dose_3D[:, :, z_iso]  # column dose_3D.shape[0] x row dose_3D.shape[1] (y,z)

    """

    def __init__(self, dose_3d):
        xx, yy, zz = dose_3d.grid
        u = dose_3d.values
        self.x = xx[::-1]
        self.y = -yy  # inverted y axis to pixel Position
        self.z = zz

        self.data = u
        self.fig = plt.figure(1, (20, 7))
        self.fig.suptitle(' ---- 3D slices viewer ----')
        self.ax1 = self.fig.add_subplot(131, aspect='equal')
        self.ax1.set_title('X - Sagital')
        self.ax2 = self.fig.add_subplot(132, aspect='equal')
        self.ax2.set_title('Y - Coronal')
        self.ax3 = self.fig.add_subplot(133, aspect='equal')
        self.ax3.set_title('Z - Axial')
        self.xplot_zline = self.ax1.axhline(color='m', linestyle='--', lw=1)
        self.xplot_zline.set_ydata(self.z[0])

        self.xplot_yline = self.ax1.axvline(color='m', linestyle='--', lw=1)
        self.xplot_yline.set_xdata(self.y[0])

        self.yplot_xline = self.ax2.axvline(color='m', linestyle='--', lw=1)
        self.yplot_xline.set_xdata(self.x[0])

        self.yplot_zline = self.ax2.axhline(color='m', linestyle='--', lw=1)
        self.yplot_zline.set_ydata(self.z[0])

        self.zplot_xline = self.ax3.axvline(color='m', linestyle='--', lw=1)
        self.zplot_xline.set_xdata(self.x[0])

        self.zplot_yline = self.ax3.axhline(color='m', linestyle='--', lw=1)
        self.zplot_yline.set_ydata(self.y[0])

        vmin = u.min()
        vmax = u.max()

        zl, yl, xl = int(u.shape[0] / 2), int(u.shape[1] / 2), int(
            u.shape[2] / 2)

        self.xslice = self.ax1.imshow(
            np.flipud(u[:, :, xl]),
            extent=(self.y[0], self.y[-1], self.z[0], self.z[-1]),
            vmin=vmin,
            vmax=vmax)

        self.yslice = self.ax2.imshow(
            np.flipud(u[:, yl, :]),
            extent=(self.x[0], self.x[-1], self.z[0], self.z[-1]),
            vmin=vmin,
            vmax=vmax)

        self.zslice = self.ax3.imshow(
            u[zl, :, :],
            extent=(self.x[0], self.x[-1], self.y[-1], self.y[0]),
            vmin=vmin,
            vmax=vmax)

        # Create and initialize x-slider
        self.sliderax1 = self.fig.add_axes([0.125, 0.08, 0.225, 0.03])
        self.sliderx = DiscreteSlider(
            self.sliderax1,
            'x-slice',
            0,
            len(self.x) - 1,
            increment=1,
            valinit=0)
        self.sliderx.on_changed(self.update_x)
        self.sliderx.set_val(xl)

        # Create and initialize y-slider
        self.sliderax2 = self.fig.add_axes([0.4, 0.08, 0.225, 0.03])
        self.slidery = DiscreteSlider(
            self.sliderax2,
            'y-slice',
            0,
            len(self.y) - 1,
            increment=1,
            valinit=0)
        self.slidery.on_changed(self.update_y)
        self.slidery.set_val(yl)

        # Create and initialize z-slider
        self.sliderax3 = self.fig.add_axes([0.675, 0.08, 0.225, 0.03])
        self.sliderz = DiscreteSlider(
            self.sliderax3,
            'z-slice',
            0,
            len(self.z) - 1,
            increment=1,
            valinit=0)
        self.sliderz.on_changed(self.update_z)
        self.sliderz.set_val(zl)

    def update_x(self, value):
        value = int(value)
        self.xslice.set_data(np.flipud(self.data[:, :, value]))
        self.yplot_xline.set_xdata(self.x[value])
        self.zplot_xline.set_xdata(self.x[value])

    def update_y(self, value):
        value = int(value)
        self.yslice.set_data(np.flipud(self.data[:, value, :]))
        self.xplot_yline.set_xdata(self.y[value])
        self.zplot_yline.set_ydata(self.y[value])

    def update_z(self, value):
        value = int(value)
        self.zslice.set_data(self.data[value, :, :])
        self.xplot_zline.set_ydata(self.z[value])
        self.yplot_zline.set_ydata(self.z[value])

    def xlabel(self, *args, **kwargs):
        self.ax2.set_xlabel(*args, **kwargs)
        self.ax3.set_xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        self.ax1.set_xlabel(*args, **kwargs)
        self.ax3.set_ylabel(*args, **kwargs)

    def zlabel(self, *args, **kwargs):
        self.ax1.set_ylabel(*args, **kwargs)
        self.ax2.set_ylabel(*args, **kwargs)

    def show(self):
        figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.show()
