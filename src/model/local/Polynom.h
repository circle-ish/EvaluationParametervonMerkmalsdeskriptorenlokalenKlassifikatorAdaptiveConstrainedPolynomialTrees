#ifndef POLYNOM_H
#define	POLYNOM_H

#include <iostream>
#include <log4cxx/logger.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "model/Types.h"

namespace faceAnalysis {
  class Polynom {
    static log4cxx::LoggerPtr logger;

  public:
    explicit Polynom(const ClassifierType classifierType, const bool sort_polynom_,
                     const int id);
    explicit Polynom(const std::string path);
    Polynom() {}
    ~Polynom() {}

    void InitAnswer(const int rows, const int cols);
    void GenerateSumAnswer();
    double GenerateAnswer(
            const std::vector<double> &feature_map_thrd_vector,
            const int row_id,
            const int col_id
            );
    void DeleteAnswer();
    void DrawAnswerMap();
    double answer_map_element(const int row_id, const int col_id);

    void Train(const cv::Mat c, const cv::Mat y);
    void Update(
            std::vector<double> &feature_map_thrd_vector,
            const double y,
            const double alpha
            );

    void Load(const std::string path);
    void Save(const std::string path);

    double error() const { return error_; }
    int classifierType() const { return classifierType_; }
    int id() const { return id_; }

    /** returns the location of the maximal answer value,
     * which is calculated as a sum over a region of answer_map_ */
    cv::Point2i maxValuePosition(double &max, double &min);
    cv::Point2i maxValuePositionBySum(double &maxVal, double &minVal);
    /** convenience overload */
    cv::Point2i maxValuePosition();
    cv::Point2i maxValuePositionBySum();
    bool is_initialized() const;

  private:
    double CalculateValue(cv::Mat c);
    cv::Mat CreateX(cv::Mat c_row);
    double GetXSize(double size_of_c);

    ClassifierType classifierType_;
    bool sort_polynom_;
    int id_;
    bool is_initialized_;
    cv::Mat A_;
    cv::Mat answer_map_;
    cv::Mat answer_sum_map_;
    cv::Mat v_list_;
    cv::Mat init_M_inverted_;
    double error_;
  };
}
#endif	/* POLYNOM_H */

