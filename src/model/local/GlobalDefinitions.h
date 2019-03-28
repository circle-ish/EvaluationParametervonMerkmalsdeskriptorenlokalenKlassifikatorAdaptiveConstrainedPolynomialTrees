#ifndef GLOBALDEFINITIONS_H
#define GLOBALDEFINITIONS_H

#include <opencv2/core/core.hpp>

namespace faceAnalysis {
    /** holds the sum of FeatureType enum values */
      extern int useFeature;
    /** file where evaluation results are written in */
      extern std::string outputFile;
    /** determines the size of the region from where features are
     * extracted
     * PATCH = 2 * PATCH_RADIUS */
      extern int PATCH_RADIUS;
      extern int PATCH;
    /** if the PATCH has an even size PATCH_LEFT_HALF is PATCH_RADIUS - 1
     * else PATCH_RADIUS; this way the feature vector for a PATCH
     * always belongs to the point at PATCH_LEFT_HALF + 1,
     * no matter wether PATCH is even or odd */
      extern int PATCH_LEFT_HALF;
    /** apply PCA to feature vector */
      extern bool APPLY_PCA;
    /** percentage of retained dimensions during pca */
      extern double retainVal;

    /** make only one call to call() with default values */
      extern bool call;

    /** values to start for-loops in *wrapper with:
     * p = PATCH_RADIUS
     * a = cellSize; b = blockSize; c = d = e =
     * a1 = cellSize.height; b1 = cellSize.width */
      extern int p, a, b, a1, b1, c, d, e;
    /** for hog, cohog */
      extern cv::Size cellSize;
      extern cv::Size blockSize;
      extern cv::Size blockOverlap;
      extern int binNo;
      extern bool useSign;
      extern int offsetNo;
    /** for HOG() method: use cohog */
      extern bool cohog;

    /** for mrlbp */
      extern int ringNo;
      extern bool half;

    /** verbosity levels:
     * 0 - near silent
     * 4 - show intermediate results
     * 5 - tell me everything */
      extern int verbosity;

      extern int no;
}

#endif // GLOBALDEFINITIONS_H
