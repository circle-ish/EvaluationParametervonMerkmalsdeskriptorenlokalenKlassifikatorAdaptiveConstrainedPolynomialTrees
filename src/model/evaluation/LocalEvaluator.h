#ifndef LOCALEVALUATOR_H
#define LOCALEVALUATOR_H

#include <opencv2/highgui/highgui.hpp>

#include "Evaluator.h"
#include "model/local/LocalPolynoms.h"

namespace faceAnalysis {

class LocalEvaluator : public Evaluator {
      static log4cxx::LoggerPtr logger;

public:
      LocalEvaluator();
      virtual ~LocalEvaluator();

      virtual double Evaluate(
              const std::vector<int> &landmarks_to_evaluate
              );
      double Evaluate(
              const std::vector<int> &landmarks_to_evaluate,
              std::vector<double> &hitMissRate,
              std::vector<double> &hitMissRateSum
              );

      /** calls Evaluate() with different parameters */
      void EvaluateWrapper(
              const std::vector<int> &landmarks_to_evaluate
              );
      void call(
              const std::vector<int> &landmarks_to_evaluate
              );
};
}

#endif // LOCALEVALUATOR_H
