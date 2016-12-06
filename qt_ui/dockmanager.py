from PySide import QtCore
from PySide import QtGui

from film2dose.qt_ui.dock import Dock


#

class DockManager(QtCore.QObject):
    def __init__(self, parent, shared_widget):
        QtCore.QObject.__init__(self, parent)

        # the parent must of the QMainWindow so that docks are created as children of it
        assert (isinstance(parent, QtGui.QMainWindow))

        self.docks = []
        self.shared_widget = shared_widget

    # slot
    def new_dock(self, tp):
        # the dock objectName is unique
        docknames = [dock.objectName() for dock in self.docks]
        dockindexes = [int(str(name).partition(' ')[-1]) for name in docknames]
        if len(dockindexes) == 0:
            index = 1
        else:
            index = max(dockindexes) + 1
        name = "Window %d" % index
        new_dock = Dock(self.parent(), self.shared_widget, name, type=tp)
        self.parent().addDockWidget(QtCore.Qt.TopDockWidgetArea, new_dock)

        self.docks += [new_dock]

        # slot

    def close_dock(self, dock):
        self.docks.remove(dock)

