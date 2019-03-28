#include "model/local/Gauss.h"

log4cxx::LoggerPtr faceAnalysis::Gauss::logger(
    log4cxx::Logger::getLogger("CPT.model.Gauss"));

void faceAnalysis::Gauss::initGauss(const int size)
{
  cv::Mat gauss_vec(size, 1, CV_64F);
  size_ = size-1;

  for (int i = 0; i <= size; i++)
    {
      gauss_vec.at<double>(i, 0) = PascalTri(size, i);
    }

  gauss_ = gauss_vec * gauss_vec.t();
  int maxValue = std::numeric_limits<int>::min();
  for (int i = 0; i < gauss_.rows; i++)
    {
      for (int j = 0; j < gauss_.rows; j++)
        {
          if (gauss_.at<double>(i, j) > maxValue)
            {
              maxValue = gauss_.at<double>(i, j);
            }
        }
    }

  gauss_ /= maxValue;
}

int faceAnalysis::Gauss::PascalTri(int x, int y) const {
  if (x * y == 0 || x == y)
    return 1;
  else
    return (PascalTri(x - 1, y - 1) + PascalTri(x - 1, y));
}

double faceAnalysis::Gauss::gauss2D(const int x, const int y) const {
  return gauss_.at<double>(x + size_ / 2, y + size_ / 2);
}
