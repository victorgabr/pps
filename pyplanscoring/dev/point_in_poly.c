
//  Globals which should be set before calling this function:
//
//  int    polyCorners  =  how many corners the polygon has (no repeats)
//  float  polyX[]      =  horizontal coordinates of corners
//  float  polyY[]      =  vertical coordinates of corners
//  float  x, y         =  point to be tested
//
//  (Globals are used in this example for purposes of speed.  Change as
//  desired.)
//
//  The function will return YES if the point x,y is inside the polygon, or
//  NO if it is not.  If the point is exactly on the edge of the polygon,
//  then the function may return YES or NO.
//
//  Note that division by zero is avoided because the division is protected
//  by the "if" clause which surrounds it.

bool pointInPolygon() {

  int   i, j=polyCorners-1 ;
  bool  oddNodes=NO      ;

  for (i=0; i<polyCorners; i++) {
    if (polyY[i]<y && polyY[j]>=y
    ||  polyY[j]<y && polyY[i]>=y) {
      if (polyX[i]+(y-polyY[i])/(polyY[j]-polyY[i])*(polyX[j]-polyX[i])<x) {
        oddNodes=!oddNodes; }}
    j=i; }

  return oddNodes; }







//  public-domain code by Darel Rex Finley, 2007



int  nodes, nodeX[MAX_POLY_CORNERS], pixelX, pixelY, i, j, swap ;

//  Loop through the rows of the image.
for (pixelY=IMAGE_TOP; pixelY<IMAGE_BOT; pixelY++) {

  //  Build a list of nodes.
  nodes=0; j=polyCorners-1;
  for (i=0; i<polyCorners; i++) {
    if (polyY[i]<(double) pixelY && polyY[j]>=(double) pixelY
    ||  polyY[j]<(double) pixelY && polyY[i]>=(double) pixelY) {
      nodeX[nodes++]=(int) (polyX[i]+(pixelY-polyY[i])/(polyY[j]-polyY[i])
      *(polyX[j]-polyX[i])); }
    j=i; }

  //  Sort the nodes, via a simple “Bubble” sort.
  i=0;
  while (i<nodes-1) {
    if (nodeX[i]>nodeX[i+1]) {
      swap=nodeX[i]; nodeX[i]=nodeX[i+1]; nodeX[i+1]=swap; if (i) i--; }
    else {
      i++; }}

  //  Fill the pixels between node pairs.
  for (i=0; i<nodes; i+=2) {
    if   (nodeX[i  ]>=IMAGE_RIGHT) break;
    if   (nodeX[i+1]> IMAGE_LEFT ) {
      if (nodeX[i  ]< IMAGE_LEFT ) nodeX[i  ]=IMAGE_LEFT ;
      if (nodeX[i+1]> IMAGE_RIGHT) nodeX[i+1]=IMAGE_RIGHT;
      for (pixelX=nodeX[i]; pixelX<nodeX[i+1]; pixelX++) fillPixel(pixelX,pixelY); }}}