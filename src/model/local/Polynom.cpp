#include "model/local/Polynom.h"
#include "model/local/GlobalDefinitions.h"
#include "model/GlobalConstants.h"
#include "model/local/Gauss.h"

log4cxx::LoggerPtr faceAnalysis::Polynom::logger(
    log4cxx::Logger::getLogger("CPT.models.training.polynom"));

faceAnalysis::Polynom::Polynom(
        const ClassifierType classifierType,
        const bool sortPoly,
        const int id
        )
  : classifierType_(classifierType),
    sort_polynom_(sortPoly),
    id_(id) {
}

faceAnalysis::Polynom::Polynom(const std::string path)
  :  id_(-1),
    is_initialized_(false) {

  Load(path);
}

double faceAnalysis::Polynom::CalculateValue(cv::Mat x) {
  cv::Mat return_value;

  if (sort_polynom_) {
      cv::Mat x_after_v_list(v_list_.rows, 1, x.type());
      for (int i = 0; i < v_list_.rows; i++) {
          //TODO ist das so richtig?!!?!?!?
          if (v_list_.at<double>(i) == 0) {
              continue;
          }

          x_after_v_list.at<double>(0, i) =
              x.at<double>(0, v_list_.at<double>(i));
      }

      return_value = A_.t() * x_after_v_list;
    } else
    {
      return_value = A_.t() * x;
    }

  return return_value.at<double>(0, 0);
}

void faceAnalysis::Polynom::Save(const std::string path) {
   if (verbosity > 3) LOG4CXX_INFO(logger, "save polynom to " << path);

  cv::FileStorage file(path, cv::FileStorage::WRITE);

  if (file.isOpened())
    {
      file << "sortPoly" << sort_polynom_;
      file << "error" << error_;
      file << "classifierType" << classifierType_;
      file << "A" << A_;
      file << "initMInv" << init_M_inverted_;
      if (sort_polynom_)
        {
          file << "vList" << v_list_;
        }
      file << "id" << id_;
      file.release();
    } else
    {
      LOG4CXX_ERROR(logger, "Cannot open file with path: " << path
                    << " -- EXIT");
      exit(-1);
    }
}

void faceAnalysis::Polynom::Load(const std::string path) {
  if (verbosity > 3) LOG4CXX_INFO(logger, "load polynom from " << path);

  cv::FileStorage file(path, cv::FileStorage::READ);

  if (file.isOpened()) {
      file["sortPoly"] >> sort_polynom_;
      file["error"] >> error_;
      file["A"] >> A_;
      file["initMInv"] >> init_M_inverted_;
      if (sort_polynom_)
        {
          file["vList"] >> v_list_;
        }
      file["id"] >> id_;
      int classifier_type_integer;
      file["classifierType"] >> classifier_type_integer;
      classifierType_ = (ClassifierType) classifier_type_integer;
      file.release();
    } else
    {
      LOG4CXX_ERROR(logger, "Cannot open file with path: " << path);
    }
}

void faceAnalysis::Polynom::Train(const cv::Mat c, const cv::Mat y) {
  LOG4CXX_INFO(logger, "Train polynom");
  cv::Mat M(GetXSize(c.cols) + 1, GetXSize(c.cols) + 1, CV_64F);
  M = cv::Scalar(0);

  if (sort_polynom_)
    {
      v_list_.create(GetXSize(c.cols), 1, CV_64F);
    }
  for (int exampleCount = 0; exampleCount < c.rows; exampleCount++)
    {
      cv::Mat x = CreateX(c.row(exampleCount));

      cv::Mat xy;
      xy.create(x.rows + 1, 1, x.type());

      for (int i = 0; i < x.rows; i++)
        {
          xy.at<double>(i, 0) = x.at<double>(i, 0);
        }

      xy.at<double>(xy.rows - 1, 0) = y.at<double>(exampleCount, 0);

      M += xy * xy.t();

    }

  M /= c.rows;

  init_M_inverted_ = M.inv();

  if (sort_polynom_)
    {
      for (int v_list_count = 0; v_list_count < GetXSize(c.cols);
           v_list_count++)
        {
          v_list_.at<double>(0, v_list_count) = v_list_count;
        }
    }

  double epsilon = std::numeric_limits<double>::epsilon();
  for (int i = 0; i < GetXSize(c.cols); i++) {
      if (verbosity > 5) LOG4CXX_DEBUG(logger, "------------------------------------");
      if (verbosity > 5) LOG4CXX_DEBUG(logger, "i: " << i);
      if (sort_polynom_)
        {
          double max_value = std::numeric_limits<double>::min();
          int max_value_index = std::numeric_limits<double>::min();
          for (int j = i; j < GetXSize(c.cols); j++)
            {
              double diag2 = M.at<double>(j, j);
              if (diag2 < epsilon)
                {
                  LOG4CXX_DEBUG(logger, "Kann deltaF von j=" << j
                                << " nicht berechnen, da diag2 = " << diag2);
                  continue;
                }

              double delta_f = (M.at<double>(j, M.rows - 1)
                                * M.at<double>(j, M.rows - 1)) / diag2;
              if (delta_f > max_value)
                {
                  max_value = delta_f;
                  max_value_index = j;
                }
            }

          LOG4CXX_DEBUG(logger, "MaxValue: " << max_value << " at "
                        << max_value_index);
          LOG4CXX_DEBUG(logger, "TAUSCHE " << i << " und " << max_value_index);

          cv::Mat temp2 = M.row(i).clone();
          cv::Mat temp3 = M.row(max_value_index).clone();

          for (int row_change_x = 0; row_change_x < temp2.cols;
               row_change_x++)
            {
              M.at<double>(i, row_change_x) = temp3.at<double>(0, row_change_x);
              M.at<double>(max_value_index, row_change_x) =
                  temp2.at<double>(0, row_change_x);
            }

          int zwi1 = v_list_.at<double>(0, i);
          int zwi2 = v_list_.at<double>(0, max_value_index);
          v_list_.at<double>(0, i) = zwi2;
          v_list_.at<double>(0, max_value_index) = zwi1;

          temp2 = M.col(i).clone();
          temp3 = M.col(max_value_index).clone();

          for (int col_change_y = 0; col_change_y < temp2.rows;
               col_change_y++)
            {
              M.at<double>(col_change_y, i) = temp3.at<double>(col_change_y, 0);
              M.at<double>(col_change_y, max_value_index) =
                  temp2.at<double>(col_change_y, 0);
            }
        }

      double diag1 = M.at<double>(i, i);

      if (sort_polynom_)
        {
          if (diag1 < epsilon)
            {
              LOG4CXX_DEBUG(logger, "delete index " << i);
              v_list_.at<double>(0, i) = -1;
              continue;
            }
        }

      if (verbosity > 5) LOG4CXX_DEBUG(logger, "Teile Reihe i: " << i << " durch diag1: "
                    << diag1);

      M.row(i) /= diag1;

      if (M.at<double>(i, i) - 1 < std::numeric_limits<double>::epsilon())
        {
          M.at<double>(i, i) = 1;
        }

      for (int k = 0; k < M.rows; k++)
        {
          if (i != k)
            {
              cv::Mat calculated_solution;
              calculated_solution = M.row(k) - (M.at<double>(k, i) * M.row(i));

              for (int counterCalc = 0; counterCalc < calculated_solution.cols;
                   counterCalc++)
                {
                  M.at<double>(k, counterCalc) =
                      calculated_solution.at<double>(0, counterCalc);
                }
            }
        }

      A_ = M.col(M.rows - 1).clone();
      A_ = A_(cv::Rect(0, 0, 1, A_.rows - 1));

      if ((verbosity > 3 && (GetXSize(c.cols) - 1 == i)) || verbosity > 5) LOG4CXX_INFO(logger, "current error: " << M.at<double>(M.rows - 1,
                                                             M.rows - 1));
    }

  error_ = M.at<double>(M.rows - 1, M.rows - 1);
}

void faceAnalysis::Polynom::Update(
        std::vector<double> &feature_map_thrd_vector,
        const double y,
        const double alpha
        ) {

  cv::Mat MI_inv = init_M_inverted_(cv::Rect(0, 0, init_M_inverted_.rows - 1,
                                             init_M_inverted_.cols - 1));

  void* data = &feature_map_thrd_vector.front();
  cv::Mat feature(feature_map_thrd_vector.size(), 1, CV_64F, data);
  cv::Mat A_new = A_ + (alpha * MI_inv * feature * (y - (A_.t() * feature)));
  A_ = A_new.clone();
}


cv::Mat faceAnalysis::Polynom::CreateX(cv::Mat c_row) {
  cv::Mat temp(GetXSize(c_row.cols), c_row.rows, CV_64F);

  double index_counter = 0;
  temp.at<double>(index_counter, 0) = 1;

  for (int i = 0; i < c_row.cols; i++)
    {
      index_counter++;
      temp.at<double>(index_counter, 0) = c_row.at<double>(0, i);
    }

  switch (classifierType_) {
    case POLY_LINEAR:
      // NOTHING TO ADD HERE
      break;
    case POLY_FULL_QUADRATIC:
      for (int i = 0; i < c_row.cols; i++)
        {
          for (int j = i; j < c_row.cols; j++)
            {
              index_counter++;
              temp.at<double>(index_counter, 0) =
                  c_row.at<double>(0, i) * c_row.at<double>(0, j);
            }
        }
      break;
    case POLY_ONLY_QUADRATIC_LINEAR:
      for (int i = 0; i < c_row.cols; i++)
        {

          index_counter++;
          temp.at<double>(index_counter, 0) =
              c_row.at<double>(0, i) * c_row.at<double>(0, i);
        }
      break;
    default:
      LOG4CXX_ERROR(logger, "Wrong classifier type for polynom "
                    << "classificator -- EXIT");
      exit(-1);
    }

  // TODO ist das hier ein Speicherleck?!
  return temp.clone();
}

double faceAnalysis::Polynom::GetXSize(double size_of_c) {
  double result = 0;

  switch (classifierType_) {
    case POLY_LINEAR:
      result = size_of_c + 1;
      break;
    case POLY_FULL_QUADRATIC:
      result = 1;
      for (int i = size_of_c + 2; i > size_of_c; i--)
        {
          result *= i;
        }
      for (int i = 1; i <= 2; i++)
        {
          result /= i;
        }
      break;
    case POLY_ONLY_QUADRATIC_LINEAR:
      result = 1 + size_of_c + size_of_c;
      break;
    default:
      LOG4CXX_ERROR(logger,
                    "Wrong classifier type for polynom "
                    << "classificator -- EXIT");
      exit(-1);
    }

  return result;
}
bool faceAnalysis::Polynom::is_initialized() const
{
  return is_initialized_;
}

void faceAnalysis::Polynom::GenerateSumAnswer() {
    answer_sum_map_.release();
    answer_sum_map_.create(answer_map_.size(), answer_map_.type());
    answer_sum_map_.setTo(cv::Scalar::all(DBL_MIN));
    for (int y = PATCH_LEFT_HALF; y < answer_map_.rows - PATCH_RADIUS; ++y) {
        for (int x = PATCH_LEFT_HALF; x < answer_map_.cols - PATCH_RADIUS; ++x) {
            const int leftY = std::min(y - PATCH_LEFT_HALF, PATCH_LEFT_HALF);
            const int leftX = std::min(x - PATCH_LEFT_HALF, PATCH_LEFT_HALF);
            const int height = std::min(answer_map_.rows - PATCH_RADIUS, y + PATCH_RADIUS + 1) - (y - leftY);
            const int width = std::min(answer_map_.cols - PATCH_RADIUS, x + PATCH_RADIUS + 1) - (x - leftX);

            cv::Scalar sum = cv::sum(answer_map_(cv::Rect(x - leftX, y - leftY, width, height)));
//            std::cout << "sum=" << sum[0] << ", area=" << width * height;
            sum[0] /=  (double) (width * height);
            if (sum[0] == 0) sum[0] = DBL_MIN;
//            std::cout << ", newSum=" << sum[0] << std::endl;
//            if (sum[0] > maxVal) {
//                maxVal = sum[0];
//                maxLoc = cv::Point(x, y);
//            }
//            if (sum[0] < minVal) {
//                minVal = sum[0];
//            }
            answer_sum_map_.at<double>(y, x) = sum[0];
        }
    }
//    std::cout << "no of 0 in answer_sum_map_: " << answer_sum_map_.total() - cv::countNonZero(answer_sum_map_) << std::endl;
}

double faceAnalysis::Polynom::GenerateAnswer(
        const std::vector<double> &feature_map_thrd_dim,
        const int row_id,
        const int col_id
        ) {

  double raw_value = 0;

  if ((row_id > PATCH_LEFT_HALF)
          && (row_id + PATCH_RADIUS) < (answer_map_.rows)
          && (col_id > PATCH_LEFT_HALF)
          && (col_id + PATCH_RADIUS) < (answer_map_.cols)
          ) {

      for (int k = 0; k < A_.rows; k++) {
          raw_value += feature_map_thrd_dim.at(k) * A_.at<double>(0, k);
      }

      if (raw_value > POSITIVE_VALUE) {
          raw_value = NEGATIVE_VALUE;
      }
  }

  answer_map_.at<double>(row_id, col_id) = raw_value;
  return raw_value;
}

void faceAnalysis::Polynom::InitAnswer(const int rows, const int cols) {
  if (!is_initialized_)
    {
      answer_map_ = cv::Mat(rows, cols , CV_64F, cv::Scalar(0));
      is_initialized_ = true;
    }
}

void faceAnalysis::Polynom::DeleteAnswer() {
  if (is_initialized_)
    {
      answer_map_.release();
      is_initialized_ = false;
    }
  if (verbosity > 3) LOG4CXX_INFO(logger, "Delete Polynom " << id());
}

double faceAnalysis::Polynom::answer_map_element(const int row_id, const int col_id) {
  return answer_map_.at<double>(row_id, col_id);
}

void faceAnalysis::Polynom::DrawAnswerMap() {
    double maxVal, minVal;
    cv::Point minLoc, maxLoc = maxValuePosition(maxVal, minVal);
    cv::Mat gui, gui2;

    answer_sum_map_.convertTo(gui, CV_8UC1, 255.0 / (maxVal - minVal), -255.0 * minVal / (maxVal - minVal));
    applyColorMap(gui, gui, cv::COLORMAP_HOT);
    cv::circle(gui, maxLoc, 4, cv::Scalar(255, 0, 0),2);

    cv::minMaxLoc(answer_map_, &minVal, &maxVal, &minLoc, &maxLoc);
    answer_map_.convertTo(gui2, CV_8UC1, 255.0 / (maxVal - minVal), -255.0 * minVal / (maxVal - minVal));
    applyColorMap(gui2, gui2, cv::COLORMAP_HOT);

    cv::imshow("answer_sum_map", gui);
    cv::imshow("answer_map", gui2);

//    std::stringstream str, str2;
//    str << outputFile << "/answer_sum_map" << id_ << faceAnalysis::no << ".png";
//    str2 << outputFile << "/answer_map" << id_ << faceAnalysis::no << ".png";
//    cv::Mat tmp;
//    cv::cvtColor(gui, tmp, CV_RGB2BGR);
//    cv::imwrite(str.str(), tmp);
//    cv::cvtColor(gui2, tmp, CV_RGB2BGR);
//    cv::imwrite(str2.str(), tmp);
}

cv::Point2i faceAnalysis::Polynom::maxValuePosition() {
    double a, b;
    return maxValuePosition(a, b);
}

cv::Point2i faceAnalysis::Polynom::maxValuePositionBySum() {
    double a, b;
    return maxValuePositionBySum(a, b);
}

cv::Point2i faceAnalysis::Polynom::maxValuePositionBySum(double &maxVal, double &minVal) {
    cv::Point2i maxLoc, minLoc;
    cv::minMaxLoc(answer_sum_map_, &minVal, &maxVal, &minLoc, &maxLoc);
    return maxLoc;
}

cv::Point2i faceAnalysis::Polynom::maxValuePosition(double &maxVal, double &minVal) {
    cv::Point2i maxLoc, minLoc;
    cv::minMaxLoc(answer_map_, &minVal, &maxVal, &minLoc, &maxLoc);
  return maxLoc;
}
