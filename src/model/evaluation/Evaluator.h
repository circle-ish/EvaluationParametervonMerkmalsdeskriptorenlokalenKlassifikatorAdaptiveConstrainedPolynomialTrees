#ifndef EVALUATOR_H
#define	EVALUATOR_H

#include <vector>
#include <log4cxx/logger.h>
#include <boost/property_tree/ptree.hpp>

#include "model/Types.h"

namespace faceAnalysis {
  class Evaluator {

    static log4cxx::LoggerPtr logger;

  public:
    virtual ~Evaluator() {}
    virtual double Evaluate(const std::vector<int> &landmarks_to_evaluate) = 0;

    void Init(const boost::property_tree::ptree configuration);

protected:
    boost::property_tree::ptree configuration_;
    std::string path_to_trained_model_;
    std::vector<cv::Rect> bounding_boxes_;
    DataBaseType database_to_use_;
    Dataset dataset_;

  private:
    void GenerateEvaluationData();

    std::string path_to_bounding_boxes_;
    std::string path_to_point_folder_;
    std::string path_to_images_;
    std::string database_to_use_string_;
  };
}
#endif	/* EVALUATOR_H */

