import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals

from core.dicom_reader import ScoringDicomParser
from pyplanscoring.core.geometry import planes2array, get_oversampled_structure

plt.style.use('ggplot')


def plot_contours_mpl(pos, title=''):
    #
    # Make a canvas and add simple view
    #
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, zs=z, marker='.')
    ax.set_title(title)
    return fig, ax


def vispy_plot_contours(structure, title=''):
    sPlanes = structure['planes']
    pos, z_axis = planes2array(sPlanes)
    canvas = vispy.scene.SceneCanvas(title=title, keys='interactive', show=True, dpi=100)
    view = canvas.central_widget.add_view()

    scatter = visuals.Line(method='gl')
    scatter.set_data(pos / 10, width=3)
    # scatter = visuals.Isosurface()
    # scatter.set_data(pos)

    view.add(scatter)

    # # view.camera = 'arcball'  # or try 'arcball'
    view.camera = 'turntable'
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    axis.set_data(pos.mean(axis=0))
    vispy.app.run()


if __name__ == '__main__':
    rs = r'C:\Users\vgalves\Dropbox\Plan_Competition_Project\Competition_2017\plans\plans\Victor Alves 3180\RS.2017-PlanComp.dcm'

    # getting dicom data
    rs_obj = ScoringDicomParser(filename=rs)
    structures = rs_obj.GetStructures()

    structure = structures[7]
    sPlanes = structure['planes']
    pos, z_axis = planes2array(sPlanes)
    ov_str = get_oversampled_structure(structure, 1)

    # vispy_plot_contours(structure, structure['name'])
    vispy_plot_contours(ov_str, structure['name'] + ' UP-SAMPLED - 1 mm')



    # plot_contours(point_cloud)
    #
