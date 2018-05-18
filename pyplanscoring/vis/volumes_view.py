import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from pyplanscoring.core.geometry import planes2array, get_oversampled_structure


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


def vispy_plot_contours(pos, title=''):
    canvas = vispy.scene.SceneCanvas(title=title, keys='interactive', show=True, dpi=100)
    view = canvas.central_widget.add_view()

    # scatter = visuals.Line(method='gl')
    # scatter.set_data(pos, width=3)
    scatter = visuals.Isosurface()
    scatter.set_data(pos)

    view.add(scatter)

    # # view.camera = 'arcball'  # or try 'arcball'
    view.camera = 'turntable'
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    axis.set_data(pos.mean(axis=0))
    vispy.app.run()
