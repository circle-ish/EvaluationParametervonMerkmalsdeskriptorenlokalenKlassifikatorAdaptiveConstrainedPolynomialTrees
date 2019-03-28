#ifndef GAUSS_H
#define	GAUSS_H

#include <opencv2/core/core.hpp>
#include <log4cxx/logger.h>

namespace faceAnalysis {
  class Gauss {
    static log4cxx::LoggerPtr logger;
  public:
    Gauss() {}
    ~Gauss() {}

    void initGauss(const int size);

    double gauss2D(const int x, const int y) const ;

  private:
    int PascalTri(const int x, const int y) const ;

    cv::Mat gauss_;
    int size_;
  };
}

#endif	/* GAUSS_H */

