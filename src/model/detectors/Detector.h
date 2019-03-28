#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <vector>
#include <map>

#include <boost/shared_ptr.hpp>

#include <opencv2/imgproc/imgproc.hpp>

namespace faceAnalysis {
  class Detector {
  public:
    virtual std::map<int, cv::Rect> Detect(const cv::Mat& image) = 0;

    virtual void set_multiple_faces(const bool multiple) = 0;

    virtual void set_scale(const double scale) = 0;

    virtual double scale() = 0;

    virtual ~Detector() {
    };
  };

  typedef boost::shared_ptr<Detector> DetectorPtr;


}

#endif /* DETECTOR_H_ */
