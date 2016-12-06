from PySide import QtGui

from film2dose.qt_ui.Film2DoseWidgets import OptimizedDoseWidget, TPSWidget, PicketFenceWidget, \
    FitWidget, EditImageWidget


# from PySide.QtCore import QLocale
#
# QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
#

from PySide import QtCore


# current_locale = QtCore.QLocale()
# print(current_locale.name())


class Dock(QtGui.QDockWidget):
    def __init__(self, parent, shared_widget, name, type=0):
        QtGui.QDockWidget.__init__(self, name, parent)

        self.setObjectName(name)

        self.sharedWidget = shared_widget

        self.dockwidget = QtGui.QWidget(self)
        self.layout = QtGui.QVBoxLayout(self.dockwidget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.dockwidget.setLayout(self.layout)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Policy(5), QtGui.QSizePolicy.Policy(5))
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.dockwidget.setSizePolicy(sizePolicy)

        self.setWidget(self.dockwidget)

        self.image_widget = None
        self.widget_select(type)

    def closeEvent(self, event):
        self.parent().dockmanager.close_dock(self)

    # Analysis tool
    def widget_select(self, item):
        if self.image_widget is not None:
            self.image_widget.close()
            self.image_widget.deleteLater()

        self.type = item

        if item is 0:
            self.image_widget = EditImageWidget(self)
            self.image_widget.read_image()
            self.image_widget.show_image()
            self.layout.addWidget(self.image_widget)
        elif item is 1:
            self.image_widget = TPSWidget(self)
            self.image_widget.read_image()
            self.image_widget.show_image()
            self.layout.addWidget(self.image_widget)
        elif item is 2:
            self.image_widget = OptimizedDoseWidget(self)
            self.image_widget.read_image()
            self.image_widget.show_image()
            self.layout.addWidget(self.image_widget)
        elif item is 3:
            self.image_widget = PicketFenceWidget(self)
            self.layout.addWidget(self.image_widget)
        elif item is 4:
            self.image_widget = FitWidget(self)
            self.layout.addWidget(self.image_widget)
        elif item is 5:
            self.image_widget = GammaCompWidget(self)
            self.layout.addWidget(self.image_widget)

    def update(self):
        if self.image_widget is not None:
            self.image_widget.update()

    def pause(self):
        if self.image_widget is not None:
            self.image_widget.pause()

    def restart(self):
        if self.image_widget is not None:
            self.image_widget.restart()

    # slot
    def settings_slot(self, checked):
        self.image_widget.settings_called(checked)

    # method
    def saveState(self, settings):
        settings.setValue("type", self.type)
        self.image_widget.saveState(settings)

    # method
    def restoreState(self, settings):
        (type, ok) = settings.value("type", 0).toInt()
        self.widget_select(type)
        self.image_widget.restoreState(settings)
