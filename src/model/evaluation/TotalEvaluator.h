#ifndef TOTALEVALUATOR_H
#define TOTALEVALUATOR_H

#include "Evaluator.h"

#include <log4cxx/logger.h>
#include <boost/property_tree/ptree.hpp>

namespace faceAnalysis {
  class TotalEvaluator : public Evaluator
  {
    static log4cxx::LoggerPtr logger;

  public:
    TotalEvaluator();
    virtual ~TotalEvaluator();

    virtual double Evaluate(const std::vector<int> landmarks_to_evaluate = std::vector<int>());

  private:
    int MapToFranckDataset(const int point_id);
  };
}


#endif // TOTALEVALUATOR_H
