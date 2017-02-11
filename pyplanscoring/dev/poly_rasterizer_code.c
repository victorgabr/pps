int nodes, nodeX[MAX_POLY_CORNERS], pixelX, pixelY, i, j, swap;

//  Loop through the rows of the image.
for (pixelY = IMAGE_TOP; pixelY < IMAGE_BOT; pixelY++) {

    //  Build a list of nodes.
    nodes = 0;
    j = polyCorners - 1;
    for (i = 0; i < polyCorners; i++) {
        if (polyY[i] < (double) pixelY && polyY[j] >= (double) pixelY || polyY[j] < (double) pixelY && polyY[i] >= (double) pixelY) {
            nodeX[nodes++] = (int)(polyX[i] + (pixelY - polyY[i]) / (polyY[j] - polyY[i]) * (polyX[j] - polyX[i]));
        }
        j = i;
    }

    //  Sort the nodes, via a simple “Bubble” sort.
    i = 0;
    while (i < nodes - 1) {
        if (nodeX[i] > nodeX[i + 1]) {
            swap = nodeX[i];
            nodeX[i] = nodeX[i + 1];
            nodeX[i + 1] = swap;
            if (i) i--;
        } else {
            i++;
        }
    }

    //  Fill the pixels between node pairs.
    for (i = 0; i < nodes; i += 2) {
        if (nodeX[i] >= IMAGE_RIGHT) break;
        if (nodeX[i + 1] > IMAGE_LEFT) {
            if (nodeX[i] < IMAGE_LEFT) nodeX[i] = IMAGE_LEFT;
            if (nodeX[i + 1] > IMAGE_RIGHT) nodeX[i + 1] = IMAGE_RIGHT;
            for (pixelX = nodeX[i]; pixelX < nodeX[i + 1]; pixelX++) fillPixel(pixelX, pixelY);
        }
    }
}


// Based on public-domain code by Darel Rex Finley, 2007
// http://alienryderflex.com/polygon_fill/
void drawPolygonScan(int * polyX, int * polyY, int npoints, Common::Rect & bbox, int color, void( * plotProc)(int, int, int, void * ), void * data) {
  int * nodeX = (int * ) calloc(npoints, sizeof(int));
  int i, j;

  //  Loop through the rows of the image.
  for (int pixelY = bbox.top; pixelY < bbox.bottom; pixelY++) {
    //  Build a list of nodes.
    int nodes = 0;
    j = npoints - 1;

    for (i = 0; i < npoints; i++) {
      if ((polyY[i] < pixelY && polyY[j] >= pixelY) || (polyY[j] < pixelY && polyY[i] >= pixelY)) {
        nodeX[nodes++] = (int)(polyX[i] + (double)(pixelY - polyY[i]) / (double)(polyY[j] - polyY[i]) *
          (double)(polyX[j] - polyX[i]) + 0.5);
      }
      j = i;
    }

    //  Sort the nodes
    Common::sort(nodeX, & nodeX[nodes]);

    //  Fill the pixels between node pairs.
    for (i = 0; i < nodes; i += 2) {
      if (nodeX[i] >= bbox.right)
        break;
      if (nodeX[i + 1] > bbox.left) {
        nodeX[i] = MAX < int16 > (nodeX[i], bbox.left);
        nodeX[i + 1] = MIN < int16 > (nodeX[i + 1], bbox.right);

        drawHLine(nodeX[i], nodeX[i + 1], pixelY, color, plotProc, data);
      }
    }
  }

  free(nodeX);
}
}



void drawHLine(int x1, int x2, int y, int color, void( * plotProc)(int, int, int, void * ), void * data) {
    if (x1 > x2)
        SWAP(x1, x2);

    for (int x = x1; x <= x2; x++)
        ( * plotProc)(x, y, color, data);
}


//  public-domain code by Darel Rex Finley, 2007

int nodes, nodeX[MAX_POLY_CORNERS], pixelX, pixelY, i, j, swap;

//  Loop through the rows of the image.
for (pixelY = IMAGE_TOP; pixelY < IMAGE_BOT; pixelY++) {

    //  Build a list of nodes.
    nodes = 0;
    j = polyCorners - 1;
    for (i = 0; i < polyCorners; i++) {
        if (polyY[i] < (double) pixelY && polyY[j] >= (double) pixelY || polyY[j] < (double) pixelY && polyY[i] >= (double) pixelY) {
            nodeX[nodes++] = (int)(polyX[i] + (pixelY - polyY[i]) / (polyY[j] - polyY[i]) * (polyX[j] - polyX[i]));
        }
        j = i;
    }

    //  Sort the nodes, via a simple “Bubble” sort.
    i = 0;
    while (i < nodes - 1) {
        if (nodeX[i] > nodeX[i + 1]) {
            swap = nodeX[i];
            nodeX[i] = nodeX[i + 1];
            nodeX[i + 1] = swap;
            if (i) i--;
        } else {
            i++;
        }
    }

    //  Fill the pixels between node pairs.
    //  Code modified by SoloBold 27 Oct 2008
    //  The flagPixel method below will flag a pixel as a possible choice.
    for (i = 0; i < nodes; i += 2) {
        if (nodeX[i] >= IMAGE_RIGHT) break;
        if (nodeX[i + 1] > IMAGE_LEFT) {
            if (nodeX[i] < IMAGE_LEFT) nodeX[i] = IMAGE_LEFT;
            if (nodeX[i + 1] > IMAGE_RIGHT) nodeX[i + 1] = IMAGE_RIGHT;
            for (j = nodeX[i]; j < nodeX[i + 1]; j++) flagPixel(j, pixelY);
        }
    }
}

// TODO pick a flagged pixel randomly and fill it, then remove it from the list.
// Repeat until no flagged pixels remain.

