#include "ViolaLienhartDetector.h"

#include <opencv2/imgproc/imgproc.hpp>


log4cxx::LoggerPtr faceAnalysis::ViolaLienhartDetector::logger(
    log4cxx::Logger::getLogger("CPT.models.detectors.violaLienhartDetector"));

faceAnalysis::ViolaLienhartDetector::ViolaLienhartDetector(
    std::string path_to_cascade) {
  if (!cascade_.load(path_to_cascade))
    {
      LOG4CXX_ERROR(logger, "Could not load classifier cascade -- EXIT");
      exit(1);
    }

  scale_ = 1;
  detect_multiple_faces_ = true;
}


std::map<int, cv::Rect> faceAnalysis::ViolaLienhartDetector::Detect(
    const cv::Mat& image) {

  cv::Mat small_image(cvRound(image.rows / scale_),
                      cvRound(image.cols / scale_), CV_8UC1);
  resize(image, small_image, small_image.size(), 0, 0, cv::INTER_LINEAR);

  int flags = 0 | CV_HAAR_SCALE_IMAGE;
  if (!detect_multiple_faces_)
    {
      flags = flags | CV_HAAR_FIND_BIGGEST_OBJECT;
    }

  cascade_.detectMultiScale(small_image, faces);
  std::map<int, cv::Rect> retFaces;
  for (unsigned int i = 0; i < faces.size(); i++)
    {
      double x = cvRound(faces[i].x * scale_);
      double y = cvRound(faces[i].y * scale_);
      double w = cvRound(faces[i].width * scale_);
      double h = cvRound(faces[i].height * scale_);
      retFaces[i] = cv::Rect(x, y, w, h);
    }
  return retFaces;
}
