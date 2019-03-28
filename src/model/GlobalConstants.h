#ifndef GLOBALCONSTANTS_H
#define GLOBALCONSTANTS_H

#include <cmath>

namespace faceAnalysis
{
  // norm the energy of a local patch to this value
  static const int ENERGY_NORM_VALUE = 10000;
  // radius of the local patch
//  static const int PATCH_RADIUS = 7; //diameter 15
//  static const int PATCH = PATCH_RADIUS * 2/* + 1*/;
  // width of the face to learn/classify
  static const double NORM_FACE_WIDTH = 200;
  // value for training for the annotated pixel
  static const double POSITIVE_VALUE = 1;
  // value for training for the random pixels
  static const double NEGATIVE_VALUE = 0;
}

#endif // GLOBALCONSTANTS_H
