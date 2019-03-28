#ifndef VIOLALIENHARTDETECTOR_H_
#define VIOLALIENHARTDETECTOR_H_

#include <string>

#include <opencv2/objdetect/objdetect.hpp>
#include <log4cxx/logger.h>

#include "Detector.h"

namespace faceAnalysis {
  class ViolaLienhartDetector : public Detector {
    static log4cxx::LoggerPtr logger;

  public:
    virtual std::map<int, cv::Rect> Detect(const cv::Mat& image);

    explicit ViolaLienhartDetector(const std::string path_to_cascade);
    ~ViolaLienhartDetector() {}

    double scale() { return scale_; }
    virtual void set_scale(const double scale) { scale_ = scale; }

    virtual void set_multiple_faces(const bool multiple)
    {
      detect_multiple_faces_ = multiple;
    }

  private:
    std::vector<cv::Rect> faces;

    cv::CascadeClassifier cascade_;
    bool detect_multiple_faces_;
    double scale_;
  };
}

#endif /* VIOLALIENHARTDETECTOR_H_ */
