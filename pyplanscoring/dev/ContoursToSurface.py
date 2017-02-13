# from __main__ import vtk, qt, ctk, slicer

import vtk


class ContoursToSurface:
    def __init__(self, parent):
        parent.title = "Contours to surface"
        parent.categories = ["Radiotherapy"]
        parent.contributors = [
            "Kyle Sunderland (Queen's), Boyeong Woo (Queen's), Csaba Pinter (Queen's), Andras Lasso (Queen's)"]
        parent.helpText = "This module generates a surface from a set of planar contours."
        parent.acknowledgementText = "This is an undergraduate project by Boyeong Woo for Queen's School of Computing, supervised by Prof. Gabor Fichtinger (Laboratory for Percutaneous Surgery). Thanks to Kevin Alexander (Cancer Centre of Southeastern Ontario) for sample data."
        self.parent = parent


# end ContoursToSurface

class ContoursToSurfaceWidget:
    def __init__(self, parent=None):
        pass
        # if not parent:
        #     self.parent = slicer.qMRMLWidget()
        #     self.parent.setLayout(qt.QVBoxLayout())
        #     self.parent.setMRMLScene(slicer.mrmlScene)
        # else:
        #     self.parent = parent
        # self.layout = self.parent.layout()
        # if not parent:
        #     self.setup()
        #     self.parent.show()

    # end __init__

    def reload(self, moduleName="ContoursToSurface"):
        """ Generic reload method for scripted modules """
        # slicer.util.reloadScriptedModule(moduleName)
        pass

    # end reload

    def setup(self):
        """ Add buttons, boxes, etc. for running the module.
          One can change this freely at his/her convenience. """
        # reload button
        # reloadButton = qt.QPushButton("Reload")
        # reloadButton.setToolTip("Reload this module.")
        # reloadButton.connect("clicked()", self.reload)
        # self.layout.addWidget(reloadButton)
        #
        # # group box
        # self.box = qt.QGroupBox()
        # self.boxLayout = qt.QFormLayout(self.box)
        # self.layout.addWidget(self.box)
        #
        # # contour nodes selector
        # self.nodesSelector = slicer.qMRMLCheckableNodeComboBox()
        # self.nodesSelector.setMRMLScene(slicer.mrmlScene)
        # self.nodesSelector.nodeTypes = (("vtkMRMLContourNode"), "")
        # self.boxLayout.addRow("Contour Nodes", self.nodesSelector)
        #
        # # ruled mode selector
        # self.modeSelector = qt.QComboBox()
        # self.modeSelector.addItem("Manual")  # the algorithm being developed here
        # self.modeSelector.addItem("PointWalk")  # from vtkRuledSurfaceFilter
        # self.modeSelector.addItem("Resample")  # from vtkRuledSurfaceFilter
        # self.boxLayout.addRow("Ruled Mode", self.modeSelector)
        #
        # # option of changing some parameters
        # self.changeButton = qt.QPushButton("Change parameters")
        # self.changeButton.connect("clicked()", self.showParameters)
        # self.boxLayout.addRow(self.changeButton)
        # # group box for parameters (hidden at first)
        # self.parameters = qt.QGroupBox("Parameters")
        # self.parametersLayout = qt.QFormLayout(self.parameters)
        # self.boxLayout.addRow(self.parameters)
        # self.parameters.hide()
        # self.useDefault = True  # variable to check whether or not to use default parameters
        # # distance factor - not used for resample mode
        # self.distanceFactor = qt.QDoubleSpinBox()
        # self.distanceFactor.setRange(1, 1000)  # any reasonable range would be fine
        # self.distanceFactor.setValue(3)  # default is 3
        # self.parametersLayout.addRow("Distance Factor", self.distanceFactor)
        # # resolutions - only used for resample mode
        # self.resolution0 = qt.QSpinBox()
        # self.resolution0.setRange(3, 1000)  # any reasonable range would be fine
        # self.parametersLayout.addRow("Resolution[0]", self.resolution0)
        # self.resolution1 = qt.QSpinBox()
        # self.resolution1.setRange(1, 10)  # any reasonable range would be fine
        # self.parametersLayout.addRow("Resolution[1]", self.resolution1)
        #
        # # run button
        # runButton = qt.QPushButton("Run")
        # runButton.connect("clicked()", self.run)
        # self.boxLayout.addRow(runButton)

    # end setup

    def showParameters(self):
        """ When "change parameters" button is clicked, the parameters are shown. """
        self.parameters.show()
        self.useDefault = False
        self.changeButton.setText("Use default parameters")
        self.changeButton.connect("clicked()", self.hideParameters)

    # end showParameters

    def hideParameters(self):
        """ When "use default parameters" button is clicked, the parameters are hidden. """
        self.parameters.hide()
        self.useDefault = True
        self.distanceFactor.setValue(3)  # default is 3
        self.changeButton.setText("Change parameters")
        self.changeButton.connect("clicked()", self.showParameters)

    # end hideParameters

    def run(self):
        """ Run the filter using the selected mode. """
        # Hide current models.
        # currentModels = slicer.util.getNodes("vtkMRMLModelDisplayNode*")
        # for model in currentModels:
        #     slicer.util.getNode(model).VisibilityOff()

        # Get the outputs.
        selectedNodes = self.nodesSelector.checkedNodes()
        for node in selectedNodes:
            input = node.GetDicomRtRoiPoints()

            mode = self.modeSelector.currentText
            if mode == "Manual":
                output = self.runManual(input)
            elif mode == "PointWalk":
                output = self.runPointWalk(input)
            else:  # Resample
                output = self.runResample(input)

            # Triangulate remnant polygons
            triangle = vtk.vtkTriangleFilter()
            triangle.SetInputData(output)
            triangle.Update()
            output = triangle.GetOutput()

            # Clear temporary lines used for processing (keep only polygons)
            emptyLines = vtk.vtkCellArray()
            output.SetLines(emptyLines)

            # Make model and add to the scene.
            # model = slicer.vtkMRMLModelNode()
            # model.SetName(node.GetName() + "_RuledSurface_" + mode)
            # display = slicer.vtkMRMLModelDisplayNode()
            # slicer.mrmlScene.AddNode(model)
            # slicer.mrmlScene.AddNode(display)
            # model.SetAndObserveDisplayNodeID(display.GetID())
            # model.SetAndObservePolyData(output)
            # display.BackfaceCullingOff()
            # display.EdgeVisibilityOn()  # show edges

            # just to make it look nicer
            # display.SetColor(node.GetDisplayNode().GetColor())

    # end run

    def runPointWalk(self, input):
        """ Pass the input through the vtkRuledSurfaceFilter using the PointWalk mode. """

        surfaceFilter = vtk.vtkRuledSurfaceFilter()

        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            surfaceFilter.SetInput(input)
        else:
            surfaceFilter.SetInputData(input)
        surfaceFilter.PassLinesOn()  # necessary
        surfaceFilter.SetRuledModeToPointWalk()

        # Set the distance factor. It is used to decide when two lines are too far apart to connect.
        # The default distance factor set by the filter is 3.
        surfaceFilter.SetDistanceFactor(self.distanceFactor.value)

        # Invoking OrientLoopsOn() causes the slicer to crash. See the report.
        # surfaceFilter.OrientLoopsOn()

        surfaceFilter.Update()
        return surfaceFilter.GetOutput()

    # end runPointWalk

    def runResample(self, input):
        """ Pass the input through the vtkRuledSurfaceFilter using the Resample mode. """
        surfaceFilter = vtk.vtkRuledSurfaceFilter()

        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            surfaceFilter.SetInput(input)
        else:
            surfaceFilter.SetInputData(input)
        surfaceFilter.PassLinesOn()  # necessary
        surfaceFilter.SetRuledModeToResample()

        # Set the resolutions.
        # resolution[0] defines the resolution in the direction of the parallel lines.
        # resolution[1] defines the resolution across the parallel lines.
        if self.useDefault:
            # Use the maximum number of points on a line as resolution[0].
            # Default resolution[1] is 1.
            resolution0 = 0
            for i in range(input.GetNumberOfLines()):
                n = input.GetCell(i).GetNumberOfPoints()
                if n > resolution0:
                    resolution0 = n
            surfaceFilter.SetResolution(resolution0, 1)
        else:
            # Use the values specified by the user.
            surfaceFilter.SetResolution(self.resolution0.value, self.resolution1.value)
        surfaceFilter.Update()

        # Resample mode generates triangle strips.
        # Pass through the vtkTriangleFilter.
        triangleFilter = vtk.vtkTriangleFilter()

        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            triangleFilter.SetInput(surfaceFilter.GetOutput())
        else:
            triangleFilter.SetInputData(surfaceFilter.GetOutput())

        triangleFilter.Update()
        return triangleFilter.GetOutput()

    # end runResample

    def runManual(self, originalInput):
        """ Try the manual PointWalk algorithm instead of the vtkRuledSurfaceFilter. """

        input = vtk.vtkPolyData()
        input.DeepCopy(originalInput)

        points = input.GetPoints()
        lines = input.GetLines()
        polys = vtk.vtkCellArray()  # add triangles to this

        line1 = vtk.vtkLine()  # a line in a plane
        line2 = vtk.vtkLine()  # a line in the next plane

        nlines = input.GetNumberOfLines()  # total number of lines

        # remove keyholes from the lines
        newLines = self.fixKeyholes(input, nlines, 0.1, 2)

        self.setLinesClockwise(input, newLines)

        lines = vtk.vtkCellArray()
        lines.Initialize()
        input.DeleteCells()
        for i in newLines:
            lines.InsertNextCell(i)
        input.SetLines(lines)
        input.BuildCells()

        nlines = input.GetNumberOfLines()  # total number of lines

        # Get two consecutive planes.
        p1 = 0  # pointer to first line on plane 1
        nlns1 = self.getNumLinesOnPlane(input, nlines, p1)  # number of lines on plane 1

        while p1 + nlns1 < nlines:

            p2 = p1 + nlns1  # pointer to first line on plane 2
            nlns2 = self.getNumLinesOnPlane(input, nlines, p2)  # number of lines on plane 2

            # Initialize overlaps lists. - list of list
            # Each internal list represents a line from the plane and will store the pointers to the overlapping lines.

            # overlaps for lines from plane 1
            overlaps1 = []
            for i in range(nlns1):
                overlaps1.append([])

            # overlaps for lines from plane 2
            overlaps2 = []
            for j in range(nlns2):
                overlaps2.append([])

            # Fill the overlaps lists.
            for i in range(nlns1):
                line1.DeepCopy(input.GetCell(p1 + i))
                for j in range(nlns2):
                    line2.DeepCopy(input.GetCell(p2 + j))
                    if self.overlap(line1, line2):
                        # line i from plane 1 overlaps with line j from plane 2
                        overlaps1[i].append(p2 + j)
                        overlaps2[j].append(p1 + i)

            # Go over the overlaps lists.
            for i in range(p1, p1 + nlns1):
                line1.DeepCopy(input.GetCell(i))
                pts1 = line1.GetPointIds()
                npts1 = line1.GetNumberOfPoints()

                intersects = False

                for j in overlaps1[i - p1]:  # lines on plane 2 that overlap with line i

                    line2.DeepCopy(input.GetCell(j))
                    pts2 = line2.GetPointIds()
                    npts2 = line2.GetNumberOfPoints()

                    # Deal with possible branches.
                    # TODO: Improve the method for branching.
                    # (It only works for simple branching, so try some more complicated ones.)

                    # Get the portion of line 1 that is close to line 2.
                    divided1 = self.branch(input, pts1, npts1, j, overlaps1[i - p1])
                    dpts1 = divided1.GetPointIds()
                    dnpts1 = divided1.GetNumberOfPoints()

                    # Get the portion of line 2 that is close to line 1.
                    divided2 = self.branch(input, pts2, npts2, i, overlaps2[j - p2])
                    dpts2 = divided2.GetPointIds()
                    dnpts2 = divided2.GetNumberOfPoints()

                    # Use the divided lines for triangulation.
                    if dnpts1 > 1 and dnpts2 > 1:
                        self.manualPointWalk(input, dpts1, dnpts1, dpts2, dnpts2, polys, points)

                        # end for j

            # end for i

            # Advance the pointers.
            p1 = p2
            nlns1 = nlns2

        # Triangulate all contours which are exposed.
        self.sealMesh(input, lines, polys)

        # Initialize the output data.
        output = vtk.vtkPolyData()
        output.SetPoints(points)
        output.SetLines(lines)
        # Return the output data.
        output.SetPolys(polys)
        return output

    # end runManual

    def fixKeyholes(self, input, nlines, epsilon, minSeperation):
        """ Parameters:
          - input: the input poly data
          - nlines: total number of lines in the input
          - epsilon: the threshold distance for keyholes
          - minSeperation: the minumum number of points of seperation between points
                           for keyhole consideration
        Returns an array of vtkLine that represent the current structure with keyholes
        isolated """

        fixedLines = []
        for i in range(nlines):
            fixedLines += self.fixKeyhole(input, i, epsilon, minSeperation)
        return fixedLines

    def fixKeyhole(self, input, lineIndex, epsilon, minSeperation):
        """ Parameters:
          - input: the input poly data
          - line: the index of the current line
          - epsilon: the threshold distance for keyholes
          - minSeperation: the minumum number of points of seperation between points
                           for keyhole consideration
        Returns an array of vtkLine that represents the current line with keyholes
        isolated """

        originalLine = vtk.vtkLine()
        originalLine.DeepCopy(input.GetCell(lineIndex))

        pts = originalLine.GetPoints()
        npts = originalLine.GetNumberOfPoints()

        # If the value of flags[i] is None, the point is not part of a keyhole
        # If the value of flags[i] is an integer, it represents a point that is
        # close enough that it could be considered part of a keyhole.
        flags = [None] * (npts - 1)

        for i in range(npts - 1):
            p1 = pts.GetPoint(i)

            for j in range(i + 1, npts - 1):

                # Make sure the points are not too close together on the line index-wise
                pointsOfSeperation = min(j - i, npts - 1 - j + i)
                if pointsOfSeperation > minSeperation:

                    # if the points are close together, mark both of them as part of a keyhole
                    p2 = pts.GetPoint(j)
                    distance = vtk.vtkMath().Distance2BetweenPoints(p1, p2)
                    if distance <= epsilon:
                        flags[i] = j
                        flags[j] = i

        newLines = []
        rawNewPoints = []
        finishedNewPoints = []

        currentLayer = 0

        inChannel = False

        # Loop through all of the points in the line
        for i in range(len(flags)):

            # Add a new line if neccessary
            if currentLayer == len(rawNewPoints):
                newLine = vtk.vtkLine()

                newLinePoints = newLine.GetPointIds()
                newLinePoints.Initialize()

                points = newLine.GetPoints()
                points.SetData(pts.GetData())

                newLines.append(newLine)
                rawNewPoints.append(newLinePoints)

            # If the current point is not part of a keyhole, add it to the current line
            if flags[i] == None:
                rawNewPoints[currentLayer].InsertNextId(originalLine.GetPointId(i))
                inChannel = False

            else:
                # If the current point is the start of a keyhole add the point to the line,
                # increment the layer, and start the channel.
                if flags[i] > i and not inChannel:
                    rawNewPoints[currentLayer].InsertNextId(originalLine.GetPointId(i))
                    currentLayer += 1
                    inChannel = True

                # If the current point is the end of a volume in the keyhole, add the point
                # to the line, remove the current line from the working list, deincrement
                # the layer, add the current line to the finished lines and start the,
                # channel.
                elif flags[i] < i and not inChannel:
                    rawNewPoints[currentLayer].InsertNextId(originalLine.GetPointId(i))
                    finishedNewPoints.append(rawNewPoints.pop())
                    currentLayer -= 1
                    inChannel = True

        # Add the remaining line to the finished list.
        for i in rawNewPoints:
            finishedNewPoints.append(i)

        # Seal the lines.
        for i in finishedNewPoints:
            if not i.GetNumberOfIds() == 0:
                i.InsertNextId(i.GetId(0))

        return newLines

    def setLinesClockwise(self, input, lines):
        ''' Parameters:
        - input: the input poly data
        - lines: the list of lines to be set clockwise
        Alters the lines in the list so that their points are oriented clockwise.'''

        numberOfLines = len(lines)

        for lineIndex in range(numberOfLines):
            if not self.clockwise(input, lines[lineIndex]):
                lines[lineIndex] = self.reverseLine(input, lines[lineIndex])

    # end setLinesClockwise

    def clockwise(self, input, line):
        ''' Parameters:
          - input: the input poly data
          - line: the vtkLine that is to be checked
          Returns 'True' if the specified line is oriented in a clockwise direction.
          'False' otherwise. Based on the shoelace algorithm. '''

        numberOfPoints = line.GetNumberOfPoints()

        # Calculate twice the area of the contour
        sum = 0
        for pointIndex in range(numberOfPoints - 1):
            point1 = input.GetPoint(line.GetPointId(pointIndex))
            point2 = input.GetPoint(line.GetPointId(pointIndex + 1))
            sum += (point2[0] - point1[0]) * (point2[1] + point1[1])

        # If the area is positive, the contour is clockwise,
        # if it is negative, the contour is counter-clockwise.
        return sum > 0

    # end clockwise

    def reverseLine(self, input, originalLine):
        ''' Parameters:
          - input: the input poly data
          - originalLine: the vtkLine that is to be reversed
          Returns the vtkLine that is the reverse of the original.'''

        numberOfPoints = originalLine.GetNumberOfPoints()

        newLine = vtk.vtkLine()
        newPoints = newLine.GetPointIds()
        newPoints.Initialize()

        for pointInLineIndex in list(reversed(range(numberOfPoints))):
            newPoints.InsertNextId(originalLine.GetPointId(pointInLineIndex))

        return newLine

    # end reverseLine

    # helper function for runManual
    def getNumLinesOnPlane(self, input, nlines, p):
        """ Parameters:
          - input: the input poly data
          - nlines: total number of lines in the input
          - p: pointer to the first line on the plane of interest
        Returns the number of lines on the plane of interest. """
        plane = input.GetCell(p).GetBounds()[4]  # z-value
        # Advance the pointer until z-value changes.
        i = p + 1
        while i < nlines and input.GetCell(i).GetBounds()[4] == plane:
            i += 1
        return i - p

    # end getNumLinesOnPlane

    # helper function for runManual
    def overlap(self, line1, line2):
        """ Parameters:
          - line1 and line2: the two lines of interest
        Returns true if the bounds of the two lines overlap. """
        bounds1 = line1.GetBounds()
        bounds2 = line2.GetBounds()
        # true if there are overlaps in x-value ranges and y-value ranges of the two lines
        # bounds[0] is min-x; bounds[1] is max-x; bounds[2] is min-y; bounds[3] is max-y
        return bounds1[0] < bounds2[1] and bounds1[1] > bounds2[0] and bounds1[2] < bounds2[3] and bounds1[3] > bounds2[
            2]

    # end overlap

    # helper function
    def inside(self, line1, line2):
        """ Parameters:
          - line1 and line2: the two lines of interest
        Returns true if line1 is inside line2.
        (This function is not used yet. I'm not sure how this could be used and whether this would be useful or not.) """
        bounds1 = line1.GetBounds()
        bounds2 = line2.GetBounds()
        # true if x-value ranges and y-value ranges of line1 are inside those of line2
        return bounds1[0] > bounds2[0] and bounds1[1] < bounds2[1] and bounds1[2] > bounds2[2] and bounds1[3] < bounds2[
            3]

    # end inside

    # helper function for runManual
    def branch(self, input, pts, npts, i, overlaps):
        """ Parameters:
          - input: the input poly data
          - pts and npts: the line to be divided (trunk)
          - i: pointer to the line that is to be connected to the trunk (branch)
          - overlaps: list of all the lines that overlap with the trunk (possible branches)
        Get the portion of the trunk closest to the branch of interest. """
        divided = vtk.vtkLine()
        dpts = divided.GetPointIds()
        dpts.Initialize()

        # Discard some points on the trunk so that the branch connects to only a part of the trunk.
        prev = False  # whether or not the previous point was added
        for j in range(npts):
            point = input.GetPoint(pts.GetId(j))
            # See if the point's closest branch is the input branch.
            if self.getClosestBranch(input, point, overlaps) == i:
                dpts.InsertNextId(pts.GetId(j))
                prev = True
            else:
                if prev:
                    # Add one extra point to close up the surface.
                    # (I'm not sure if this is the best way. You can change this if you want.)
                    dpts.InsertNextId(pts.GetId(j))
                prev = False
        dnpts = divided.GetNumberOfPoints()

        if dnpts > 1:
            # Determine if the trunk was originally a closed contour.
            closed = (pts.GetId(0) == pts.GetId(npts - 1))
            if closed and (dpts.GetId(0) != dpts.GetId(dnpts - 1)):
                # Make the new one a closed contour as well.
                # (I'm not sure if we have to make it closed always.)
                dpts.InsertNextId(dpts.GetId(0))

        return divided

    # end branch

    # helper function for branch
    def getClosestBranch(self, input, point, overlaps):
        """ Parameters:
          - input: the input poly data
          - point: a point on the trunk
          - overlaps: list of all the lines that overlap with the trunk
        Returns the branch closest from the point on the trunk. """
        line = vtk.vtkLine()  # a branch from the overlaps
        best = 0  # minimum distance from the point to the closest branch
        closest = 0  # pointer to the closest branch
        for i in overlaps:
            line.DeepCopy(input.GetCell(i))
            pts = line.GetPointIds()
            npts = line.GetNumberOfPoints()

            x = input.GetPoint(pts.GetId(0))  # a point from the branch
            minD2 = vtk.vtkMath().Distance2BetweenPoints(point, x)  # minimum distance from the point to the branch
            for j in range(1, npts):
                x = input.GetPoint(pts.GetId(j))
                distance2 = vtk.vtkMath().Distance2BetweenPoints(point, x)
                if distance2 < minD2:
                    minD2 = distance2

            # See if this branch is closer than the current closest.
            if best == 0 or minD2 < best:
                best = minD2
                closest = i

        return closest

    # end getClosestBranch

    # "polys" will be updated by this function (passed by reference).
    def manualPointWalk(self, input, pts1, npts1, pts2, npts2, polys, points):
        """ Parameters:
          - input: the input poly data
          - pts1 and npts1: one of the lines to be connected (line 1)
          - pts2 and npts2: one of the lines to be connected (line 2)
          - polys: at the end, polys should hold the surface comprised of triangles
        This is modified from vtkRuledSurfaceFilter::PointWalk. """

        # Pre-calculate and store the closest points.

        # closest from line 1 to line 2
        closest1 = []
        for i in range(npts1):
            point = input.GetPoint(pts1.GetId(i))
            closest1.append(self.getClosestPoint(input, point, pts2, npts2))

        # closest from line 2 to line 1
        closest2 = []
        for i in range(npts2):
            point = input.GetPoint(pts2.GetId(i))
            closest2.append(self.getClosestPoint(input, point, pts1, npts1))

        # Orient loops.
        # Use the 0th point on line 1 and the closest point on line 2.
        # (You might want to try different starting points for better results.)
        startLoop1 = 0
        startLoop2 = closest1[0]

        x = input.GetPoint(pts1.GetId(startLoop1))  # first point on line 1
        a = input.GetPoint(pts2.GetId(startLoop2))  # first point on line 2
        xa = vtk.vtkMath().Distance2BetweenPoints(x, a)

        # Determine the maximum edge length.
        # (This is just roughly following the scheme from the vtkRuledSurfaceFilter.
        # You might want to try different numbers, or maybe not use this at all.)
        distance2 = xa * self.distanceFactor.value * self.distanceFactor.value

        # Determine if the loops are closed.
        # A loop is closed if the first point is repeated as the last point.
        # (I did not yet see any example where there were open contours, but I'll leave this for now.)
        loop1Closed = (pts1.GetId(0) == pts1.GetId(npts1 - 1))
        loop2Closed = (pts2.GetId(0) == pts2.GetId(npts2 - 1))

        # Determine the ending points.
        endLoop1 = self.getEndLoop(startLoop1, npts1, loop1Closed)
        endLoop2 = self.getEndLoop(startLoop2, npts2, loop2Closed)

        # for backtracking
        left = -1
        up = 1

        # Initialize the DP table.
        # Rows represent line 1. Columns represent line 2.

        # Fill the first row.
        firstRow = [xa]
        backtrackRow = [0]
        loc2 = self.getNextLoc(startLoop2, npts2, loop2Closed)
        for j in range(1, npts2):
            p = input.GetPoint(pts2.GetId(loc2))  # current point on line 2
            # Use the distance between first point on line 1 and current point on line 2.
            xp = vtk.vtkMath().Distance2BetweenPoints(x, p)
            firstRow.append(firstRow[j - 1] + xp)
            backtrackRow.append(left)
            loc2 = self.getNextLoc(loc2, npts2, loop2Closed)

        # Fill the first column.
        score = [firstRow]  # 2D list
        backtrack = [backtrackRow]  # 2D list
        loc1 = self.getNextLoc(startLoop1, npts1, loop1Closed)
        for i in range(1, npts1):
            p = input.GetPoint(pts1.GetId(loc1))  # current point on line 1
            # Use the distance between first point on line 2 and current point on line 1
            pa = vtk.vtkMath().Distance2BetweenPoints(p, a)
            score.append([score[i - 1][0] + pa] + [0] * (npts2 - 1))  # appending another row
            backtrack.append([up] + [0] * (npts2 - 1))  # appending another row
            loc1 = self.getNextLoc(loc1, npts1, loop1Closed)

        # Fill the rest of the table.
        prev1 = startLoop1
        prev2 = startLoop2
        loc1 = self.getNextLoc(startLoop1, npts1, loop1Closed)
        loc2 = self.getNextLoc(startLoop2, npts2, loop2Closed)
        for i in range(1, npts1):
            x = input.GetPoint(pts1.GetId(loc1))  # current point on line 1
            for j in range(1, npts2):
                a = input.GetPoint(pts2.GetId(loc2))  # current point on line 2
                xa = vtk.vtkMath().Distance2BetweenPoints(x, a)

                # Use the pre-calculated closest point.
                # (This was not in the report. Sometimes it seemed like it had to take a longer span, so that's why I added this,
                # but I have not yet fully tested this. If this does not seem much better, you can remove this part.)
                if loc1 == closest2[prev2]:  # if loc1 is the closest from prev2
                    score[i][j] = score[i][j - 1] + xa
                    backtrack[i][j] = left
                elif loc2 == closest1[prev1]:  # if loc2 is the closest from prev1
                    score[i][j] = score[i - 1][j] + xa
                    backtrack[i][j] = up

                # score[i][j] = min(score[i][j-1], score[i-1][j]) + xa
                elif score[i][j - 1] <= score[i - 1][j]:
                    score[i][j] = score[i][j - 1] + xa
                    backtrack[i][j] = left
                else:
                    score[i][j] = score[i - 1][j] + xa
                    backtrack[i][j] = up

                # Advance the pointers.
                prev2 = loc2
                loc2 = self.getNextLoc(loc2, npts2, loop2Closed)
            prev1 = loc1
            loc1 = self.getNextLoc(loc1, npts1, loop1Closed)

        # Backtrack.
        loc1 = endLoop1
        loc2 = endLoop2
        while i > 0 or j > 0:
            x = input.GetPoint(pts1.GetId(loc1))  # current point on line 1
            a = input.GetPoint(pts2.GetId(loc2))  # current point on line 2
            xa = vtk.vtkMath().Distance2BetweenPoints(x, a)

            if backtrack[i][j] == left:
                prev2 = self.getPrevLoc(loc2, npts2, loop2Closed)
                b = input.GetPoint(pts2.GetId(prev2))  # previous point on line 2
                xb = vtk.vtkMath().Distance2BetweenPoints(x, b)
                # Insert triangle if the spans are not larger than the maximum distance.

                # if xa <= distance2 and xb <= distance2:\
                currentTriangle = [pts1.GetId(loc1), pts2.GetId(loc2), pts2.GetId(prev2), pts1.GetId(loc1)]
                polys.InsertNextCell(3)
                polys.InsertCellPoint(currentTriangle[0])
                polys.InsertCellPoint(currentTriangle[1])
                polys.InsertCellPoint(currentTriangle[2])

                # Advance the pointers (backwards).
                j -= 1
                loc2 = prev2

            else:  # up
                prev1 = self.getPrevLoc(loc1, npts1, loop1Closed)
                y = input.GetPoint(pts1.GetId(prev1))  # previous point on line 1
                ya = vtk.vtkMath().Distance2BetweenPoints(y, a)
                # Insert triangle if the triangle does not go out of the bounds of the contours.
                # If checkSurface returns None, the triangle is fine, otherwise it returns a list of new triangles
                # to be added instead
                currentTriangle = [pts1.GetId(loc1), pts2.GetId(loc2), pts1.GetId(prev1), pts1.GetId(loc1)]
                polys.InsertNextCell(3)
                polys.InsertCellPoint(currentTriangle[0])
                polys.InsertCellPoint(currentTriangle[1])
                polys.InsertCellPoint(currentTriangle[2])

                # Advance the pointers (backwards).
                i -= 1
                loc1 = prev1

    # TODO: Deal with rapid changes and internal contours.
    # (I was thinking it might be useful to be able to detect when a triangle goes outside the contour lines so that it can be corrected.)
    # end manualPointWalk

    def sealMesh(self, input, lines, polys):
        '''
        Parameters:
          - input: the input poly data
          - lines: vtkCellArray representing all the contours in the mesh
          - polys: vtkCellArray representing all of the polys in the mesh
        This function seals all contours that do not have polygons connecting on both sides.
        '''

        numLines = lines.GetNumberOfCells()
        numPolys = polys.GetNumberOfCells()

        # Keep track of whether polygons connect to contours from above or below.
        polygonsToBelow = [False] * numLines
        polygonsToAbove = [False] * numLines

        # Loop through the lines.
        line = vtk.vtkLine()
        for i in range(numLines - 1):
            line.DeepCopy(input.GetCell(i))
            pts = line.GetPointIds()
            z = line.GetBounds()[4]

            # Loop through the polygons
            polygonIds = vtk.vtkIdList()
            polys.SetTraversalLocation(0)
            for j in range(numPolys):

                # If polygons connect to the current contour from above and below,
                # it doesnt need to be sealed.
                if polygonsToAbove[i] and polygonsToBelow[i]:
                    break

                # Get the Id and z coordinates of the polygon corners.
                polys.GetNextCell(polygonIds)
                p1Id = polygonIds.GetId(0)
                p2Id = polygonIds.GetId(1)
                p3Id = polygonIds.GetId(2)
                z1 = input.GetPoint(p1Id)[2]
                z2 = input.GetPoint(p2Id)[2]
                z3 = input.GetPoint(p3Id)[2]

                # Only check polygons which lie on the same Z.
                if z1 == z or z2 == z or z3 == z:

                    # Check to see if the corners of the polygon lie on the line.
                    p1OnLine = self.onLine(pts, p1Id)
                    p2OnLine = self.onLine(pts, p2Id)
                    p3OnLine = self.onLine(pts, p3Id)

                    # If any of the corners of the current polygon lies on the current line.
                    if p1OnLine or p2OnLine or p3OnLine:
                        if not p1OnLine:
                            zNotOnLine = z1
                        elif not p2OnLine:
                            zNotOnLine = z2
                        elif not p3OnLine:
                            zNotOnLine = z3

                            # Check to see if the current polygon connects to the contour above or below it.
                        if zNotOnLine > z:
                            polygonsToAbove[i] = True
                        else:
                            polygonsToBelow[i] = True

        # Seal all contours which are only connected on one side.
        for i in range(numLines):
            line.DeepCopy(input.GetCell(i))
            if not (polygonsToAbove[i] and polygonsToBelow[i]):
                self.triangulateLine(input, line, polys)

    # end sealMesh

    def onLine(self, lineIds, id):
        '''
        Parameters:
        - lineIds: vtkIdList of the points in a line
        - id: id that is being checked
        Returns true if 'id' is in 'lineIds'
        '''

        numberOfPoints = lineIds.GetNumberOfIds()
        for currentId in range(numberOfPoints):
            if lineIds.GetId(currentId) == id:
                return True
        return False
        # end onLine

    def triangulateLine(self, input, line, polys):
        '''
        Parameters:
        - input: the input poly data
        - line: vtkLine representing the line to be triangulated
        - polys: vtkCellArray containing current polygons
        This function adds new polygons to polys that represent a 2D
        triangulation of 'line' on the x-y plane.
        '''

        npts = line.GetNumberOfPoints()

        linePolyData = vtk.vtkPolyData()
        linePolyData.SetPoints(line.GetPoints())

        boundary = vtk.vtkPolyData()
        boundary.SetPoints(linePolyData.GetPoints())

        # Use vtkDelaunay2D to triangulate the line
        # and produce new polyons
        delaunay = vtk.vtkDelaunay2D()

        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            delaunay.SetInput(linePolyData)
            delaunay.SetSource(boundary)
        else:
            delaunay.SetInputData(linePolyData)
            delaunay.SetSourceData(boundary)
        delaunay.Update()
        out = delaunay.GetOutput()
        newPolygons = out.GetPolys()

        # Check each new polygon to see if it is inside the line.
        ids = vtk.vtkIdList()
        for i in range(out.GetNumberOfPolys()):
            newPolygons.GetNextCell(ids)

            # Get the center of the polygon
            x = 0
            y = 0
            for i in range(3):
                point = input.GetPoint(line.GetPointId(ids.GetId(i)))
                x += point[0]
                y += point[1]
            x /= 3
            y /= 3

            # Check to see if the center of the polygon lies inside the line.
            if self.pointInsideLine(input, line, (x, y)):
                polys.InsertNextCell(3)
                polys.InsertCellPoint(line.GetPointId(ids.GetId(0)))
                polys.InsertCellPoint(line.GetPointId(ids.GetId(1)))
                polys.InsertCellPoint(line.GetPointId(ids.GetId(2)))

    # end triangulateLine

    def pointInsideLine(self, input, line, point):
        '''
        Parameters:
        - input: the input poly data
        - line: vtkLine representing the line that is being checked
        - point: an (x, y) tuple that represents the point that is being checked
        This function uses ray casting to determine if a point lies within the line
        '''

        # Create a ray that starts outside the polygon and goes to the point being checked
        bounds = line.GetBounds()
        rayPoint1 = (bounds[0] - 10, bounds[2] - 10)
        rayPoint2 = point
        ray = (rayPoint1, rayPoint2)

        # Check all of the edges to see if they intersect
        numberOfIntersections = 0
        for i in range(line.GetNumberOfPoints() - 1):
            edgePoint1 = input.GetPoint(line.GetPointId(i))
            edgePoint2 = input.GetPoint(line.GetPointId(i + 1))
            edge = (edgePoint1, edgePoint2)

            if point == edgePoint1 or point == edgePoint2:
                return True

            if self.lineIntersection(ray, edge, False):
                numberOfIntersections += 1

        # If the number of intersections is odd, the point is inside
        return numberOfIntersections % 2 == 1

    # end pointInsideLine

    def lineIntersection(self, line1, line2, countInfinite):
        '''
        Parameters:
        - input: the input poly data
        - line1: tuple of ((lx1,ly1),(lx2,ly2)) that identifies the first line segment
        - line2: tuple of ((lx1,ly1),(lx2,ly2)) that identifies the second line segment
        - countInfinite: boolean which dictates if lines that are coincident should be considered to intersect
        This function returns True if the two line segments intersect, False otherwise.
        '''

        # Get the bounding region for the intersection
        xmin = max(min(line1[0][0], line1[1][0]), min(line2[0][0], line2[1][0]))
        xmax = min(max(line1[0][0], line1[1][0]), max(line2[0][0], line2[1][0]))
        ymin = max(min(line1[0][1], line1[1][1]), min(line2[0][1], line2[1][1]))
        ymax = min(max(line1[0][1], line1[1][1]), max(line2[0][1], line2[1][1]))

        # If the two lines don't overlap, no intersection
        if xmin > xmax or ymin > ymax:
            return False

        slope1 = self.getSlope(line1)
        slope2 = self.getSlope(line2)

        intercept1 = self.getIntercept(line1, slope1)
        intercept2 = self.getIntercept(line2, slope2)

        # Lines either parallel or coincident
        if (slope1 == slope2):
            if slope1 == None:
                return countInfinite and line1[0][1] == line2[0][1]
            else:
                return countInfinite and intercept1 == intercept2

        # If one of the lines has no distance, check to see if it lies on the other line as a point
        if line1[0] == line1[1] and line2[0] == line2[1]:
            return line1[0] == line2[0]
        elif line1[0] == line1[1]:
            return self.pointOnLine(slope2, intercept2, line1[0])
        elif line2[0] == line2[1]:
            return self.pointOnLine(slope1, intercept1, line2[0])

        # Find the intersection location for the lines
        if slope1 == None:
            x = line1[0][0]
            y = slope2 * x + intercept2
        elif slope2 == None:
            x = line2[0][0]
            y = slope1 * x + intercept1
        else:
            x = (intercept2 - intercept1) / (slope1 - slope2)
            if slope1 == 0:
                y = intercept1
            elif slope2 == 0:
                y = intercept2
            else:
                y = slope1 * x + intercept1

        # If the intersection lies within the segments, return True, otherwise return False.
        if x >= xmin and \
                        x <= xmax and \
                        y >= ymin and \
                        y <= ymax:
            return True
        else:
            return False

    # end lineIntersection

    def getSlope(self, line):
        '''
        Parameters:
        - input: the input poly data
        - line: tuple of ((lx1,ly1),(lx2,ly2)) that identifies the line
        This function returns the slope of the line, or None if the slope is infinite.
        '''
        xChange = line[0][0] - line[1][0]
        if xChange == 0:
            return None
        else:
            yChange = line[0][1] - line[1][1]
            return yChange / xChange

    # end getSlope

    def getIntercept(self, edge, slope):
        '''
        Parameters:
        - input: the input poly data
        - line: tuple of ((lx1,ly1),(lx2,ly2)) that identifies the line
        This function returns the y-intercept of the line, or None if the slope is infinite.
        '''
        if slope == None:
            return None
        else:
            return edge[0][1] - slope * edge[0][0]
            # end getIntercept

    def pointOnLine(self, m, b, point):
        '''
        Parameters:
        - m: the slope of the line
        - b: the y-intercept of the line
        - point: an (x, y) tuple that represents the point that is being checked
        This function returns True if the point lies on the line, False otherwise.
        '''
        epsilon = 0.00001
        y = m * point[0] + b
        return abs(y - point[1]) < epsilon

    # end pointOnLine

    # helper function for manualPointWalk
    def getClosestPoint(self, input, point, pts, npts):
        """ Parameters:
          - input: the input poly data
          - point: a point from a line
          - pts and npts: the other line that is to be connected
        Returns the point on the given line that is closest to the given point. """
        x = input.GetPoint(pts.GetId(0))  # point from the given line
        minD2 = vtk.vtkMath().Distance2BetweenPoints(point, x)  # minimum distance from the point to the line
        closest = 0  # pointer to the closest point
        for i in range(1, npts):
            x = input.GetPoint(pts.GetId(i))
            distance2 = vtk.vtkMath().Distance2BetweenPoints(point, x)
            if distance2 < minD2:
                minD2 = distance2
                closest = i
        return closest

    # end getClosestPoint

    # helper function for manualPointWalk
    def getEndLoop(self, startLoop, npts, loopClosed):
        """ Parameters:
          - startLoop: pointer to the starting point
          - npts: number of points on the loop
          - loopClosed: whether or not the loop is closed
        Returns the ending point for the loop. """
        if startLoop != 0:
            if loopClosed:
                return startLoop
            return startLoop - 1
        # If startLoop was 0, then it doesn't matter whether or not the loop was closed.
        return npts - 1

    # end getEndLoop

    # helper function for manualPointWalk
    def getNextLoc(self, loc, npts, loopClosed):
        """ Parameters:
          - loc: pointer to the current point
          - npts: number of points on the loop
          - loopClosed: whether or not the loop is closed
        Returns the next point on the loop. """
        if loc + 1 == npts:  # if the current location is the last point
            if loopClosed:
                # Skip the repeated point.
                return 1
            return 0
        return loc + 1

    # end getNextLoc

    # helper function for manualPointWalk
    def getPrevLoc(self, loc, npts, loopClosed):
        """ Parameters:
          - loc: pointer to the current point
          - npts: number of points on the loop
          - loopClosed: whether or not the loop is closed
        Returns the previous point on the loop. """
        if loc - 1 == -1:  # if the current location is the first point
            if loopClosed:
                # Skip the repeated point.
                return npts - 2
            return npts - 1
        return loc - 1
        # end getPrevLoc

        # end ContoursToSurfaceWidget
