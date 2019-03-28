/*
 * File:   Types.h
 * Author: ttoenige
 *
 */

#ifndef TYPES_H
#define	TYPES_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <boost/graph/adjacency_list.hpp>

#define UNUSED(x) (void)(x)

/** OpenCV constants */
#define SRC_TYPE double
#define SRC_CV_TYPE CV_64FC1
//#define SRC_VEC_TYPE cv::Vec2b

#define GRAD_TYPE short
#define GRAD_CV_TYPE CV_16SC2
#define GRAD_VEC_TYPE cv::Vec2s

#define BINMAG_TYPE double
#define BINMAG_CV_TYPE CV_64FC2
#define BINMAG_VEC_TYPE cv::Vec2d

#define BIN_TYPE double
#define BIN_CV_TYPE CV_64FC1

#define MR_TYPE double
#define MR_CV_TYPE CV_64FC4
#define MR_VEC_TYPE cv::Vec4d

typedef cv::Mat PointsOfImage;
typedef std::pair<cv::Mat, PointsOfImage> DatasetExample;
typedef std::vector < DatasetExample > Dataset;

typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property < boost::edge_weight_t, double > > Graph;
typedef Graph::edge_descriptor GraphEdge;
typedef Graph::vertex_descriptor GraphVertex;

typedef boost::adjacency_list < boost::listS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property < boost::edge_weight_t, double > > Tree;
typedef Tree::edge_descriptor TreeEdge;
typedef Tree::vertex_descriptor TreeVertex;

typedef std::pair<int, int> E;

namespace faceAnalysis
{

    typedef std::vector<std::vector<cv::Mat> > MatrixOfMats_t;
    typedef std::vector<std::vector<std::vector<double> > > Mat3D_t;

    /** determines the used methods;
     * bitshift is an easier way of adding enums by simply incrementing instead of calculating powers of 2 */
     enum FeatureType {
         USE_HOG = 1 << 1,
         USE_COHOG = 1 << 2,
         USE_MRLBP = 1 << 3,
         USE_GRAD = 1 << 4,
         USE_LBP = 1 << 5
     };

     enum BinPatchingShapeType {
         SHAPE_VERTICAL,
         SHAPE_HORIZONTAL,
         SHAPE_QUADRATIC
     };

  /**
 * type of the local classifier to learn
 * (at the moment only POLY_ONLY_QUADRATIC_LINEAR is supported)
 */
  enum ClassifierType {
    SVM,
    POLY_FULL_QUADRATIC,
    POLY_LINEAR,
    POLY_ONLY_QUADRATIC_LINEAR,
    CLASSIFIER_INIT
  };

  /*
 * which database to use for training
 */
  enum DataBaseType {
    DATABASETYPE_INIT,
    MULTIPIE,
    FRANCK,
    AFW_300W,
    HELEN_TEST_300W,
    HELEN_TRAIN_300W,
    IBUG_300W,
    LFPW_TEST_300W,
    LFPW_TRAIN_300W
  };


  /**
 * which images of the database to use for training
 */
  enum ImagesToUseType {
    IMAGE_TO_USE_INIT,
    FIRST,
    ALL,
    RANDOM
  };

}


#endif	/* TYPES_H */


