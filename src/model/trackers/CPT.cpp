#include "CPT.h"

#include <limits>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp> 

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "model/dataHandlers/DatasetHandler.h"
#include "model/local/Gauss.h"
#include "model/CmdInputMapping.h"

log4cxx::LoggerPtr faceAnalysis::CPT::logger(
    log4cxx::Logger::getLogger("CPT.model.CPT"));


faceAnalysis::CPT::CPT(const boost::property_tree::ptree configuration,
                       std::vector<int> landmarksToUse,
                       std::string output_folder_path)
  : output_folder_path_(output_folder_path),
    imageSet(false) {
  first_detection_ = true;

  weight_global_ = configuration.get<int>("evaluation.weightGlobal");
  weight_local_ = configuration.get<int>("evaluation.weightLocal");

  global_model_file_name_ =
      configuration.get<std::string>("training.globalModelFileName");

  globalModel_ = GlobalTreePtr(new GlobalTree());
  localPolynoms_ = LocalPolynomsPtr(new LocalPolynoms(configuration,
                                                      landmarksToUse));
}


void faceAnalysis::CPT::ClearPoints() {
  LOG4CXX_INFO(logger, "Clear Points");
  collected_face_points_.clear();
  collected_face_points_values_.clear();
}

void faceAnalysis::CPT::Load(const std::string path_string) {
  // local
  localPolynoms_->ClearAllPolynoms();
  localPolynoms_->Load(path_string);

  // global
  std::stringstream loadGlobalPathStr;
  loadGlobalPathStr << path_string << "/" << global_model_file_name_;

  LOG4CXX_INFO(logger, loadGlobalPathStr.str());
  globalModel_->Load(loadGlobalPathStr.str());
}


void faceAnalysis::CPT::DetectFeatures(const bool reinit,
                                       const cv::Point2f rootPoint) {

  if (!imageSet) {
      LOG4CXX_ERROR(logger, "No new image set to detect features!");
    }

  imageSet = false;

  using cv::Mat;
  using cv::Rect;
  using std::vector;
  using std::numeric_limits;

  collected_face_points_.clear();
  collected_face_points_values_.clear();

  double t_all = (double) cvGetTickCount();
  // Scale / convert  face
  //    Mat scaledFace;
  int faceWidth = current_face_detection_.width;
  //    double scaleFactor = (NORM_FACE_WIDTH + 90) / (double) faceWidth;
  current_scale_factor_ = (NORM_FACE_WIDTH) / (double) faceWidth;


  resize(current_image_, scaled_whole_image_,
         cv::Size(current_scale_factor_ * current_image_.cols,
                  current_scale_factor_ * current_image_.rows));

  //    imshow("sss", scaledWhole);
  //    waitKey(0);

  Rect scaledRect;
  scaledRect.x = round(current_scale_factor_ * current_face_detection_.x);
  scaledRect.y = round(current_scale_factor_ * current_face_detection_.y);
  scaledRect.width = round(current_scale_factor_ * current_face_detection_.width);
  scaledRect.height = round(current_scale_factor_ * current_face_detection_.height);

  //    Mat detectionImg = scaledWhole(scaledRect);

  // GUI init
  gui_ = scaled_whole_image_.clone();

  double t_allPolynoms_1 = (double) cvGetTickCount();

  if (reinit)
    {
      if (!first_detection_) {
          LOG4CXX_INFO(logger, "Free arrays");
          FreeBc();
          FreeBcPosition();
          FreePolynoms();
          localPolynoms_->FreeFeatureMap();
        }

      GenerateB_c(scaled_whole_image_.rows, scaled_whole_image_.cols);
      localPolynoms_->InitFeatureMap(scaled_whole_image_.rows, scaled_whole_image_.cols);
    }

  localPolynoms_->SetCurrentImage(scaled_whole_image_);

  t_allPolynoms_1 = (double) cvGetTickCount() - t_allPolynoms_1;

  // get root vertex
  TreeVertex root = globalModel_->root();
  int rootID = (int) root;
  vector<TreeVertex> childs = globalModel_->childs(root);

  LOG4CXX_INFO(logger, "ROOT vertex: " << rootID);
  //    LOG4CXX_INFO(logger, setw(80) << setfill('-') << "-");

  // start calculation
  double minOutputValue = numeric_limits<double>::max();
  cv::Point2i minRootPos;

  double t_recursion = (double) cvGetTickCount();

  //    Point2i midPoint(round((scaledRect.x + (scaledRect.width / 2)) - (scaledRect.width / 10)), round((scaledRect.y + (scaledRect.height / 2)) + (scaledRect.height / 10)));

  cv::Point2i midPoint;


  midPoint.x = round(current_scale_factor_ * rootPoint.x);
  midPoint.y = round(current_scale_factor_ * rootPoint.y);


  if (midPoint.x <= 5 || midPoint.y <= 5)
    {
      midPoint.x = round(current_scale_factor_ * rootPoint.x);
      midPoint.y = round(current_scale_factor_ * rootPoint.y);
      std::cout << "using this one: " << midPoint << std::endl;
    }

  all_face_points_means_ = globalModel_->all_means(rootID, midPoint);

  if (midPoint.x + SURROUNDING_MIDPOINT_DETECTION_RADIUS >= scaled_whole_image_.cols
      || midPoint.y + SURROUNDING_MIDPOINT_DETECTION_RADIUS >= scaled_whole_image_.rows)
    {
      LOG4CXX_INFO(logger, "Root vertex outside detection window " << midPoint
                   << scaled_whole_image_.rows << " "
                   << scaled_whole_image_.cols);
      exit(-1);
    }


  for (
       int col_itr = midPoint.x - SURROUNDING_MIDPOINT_DETECTION_RADIUS;
       col_itr <= midPoint.x + SURROUNDING_MIDPOINT_DETECTION_RADIUS;
       ++col_itr
       ) {

      for (
           int row_itr = midPoint.y - SURROUNDING_MIDPOINT_DETECTION_RADIUS;
           row_itr <= midPoint.y + SURROUNDING_MIDPOINT_DETECTION_RADIUS;
           ++row_itr
           ) {

          if (localPolynoms_->GetElementFeatureMap(row_itr, col_itr, 0) == 0) {
              localPolynoms_->GenerateFeature(row_itr, col_itr);
          }
          // get value for root polynom
          if (!localPolynoms_->IsAnswerMapInitialized(rootID)) {
              localPolynoms_->initAnswer(rootID);
          }

          double localAnswer =1 -  localPolynoms_->getAnswer(rootID, row_itr, col_itr);
          if (localAnswer ==1) {
              localAnswer = 1 -  localPolynoms_->GenerateAnswerForPolynom(rootID, row_itr, col_itr);
          }

          double totalValueTemp = weight_local_ * localAnswer;
          bool rootBreak = false;
          for (uint i = 0; i < childs.size(); i++) {
              int childID = (int) childs.at(i);
              double childAnswer = AnswerChild(col_itr, row_itr, childs.at(i),
                                               E(rootID, childID));
              totalValueTemp += childAnswer;
              //
              //                if (totalValueTemp > minOutputValue) {
              //                    rootBreakCounter++;
              //                    rootBreak = true;
              //                    break;
              //                }
            }

          //                calculate overall minimum
          if (!rootBreak && totalValueTemp < minOutputValue) {
              minOutputValue = totalValueTemp;
              minRootPos.x = col_itr;
              minRootPos.y = row_itr;

              collected_face_points_[rootID].first = col_itr;
              collected_face_points_[rootID].second = row_itr;
              collected_face_points_values_[rootID] = localAnswer;

              bool firstCollectionIteration = true;
              for (uint i = 0; i < childs.size(); i++) {
                  int childID = (int) childs.at(i);
                  CollectPositions(firstCollectionIteration, minRootPos.x,
                                   minRootPos.y, childs.at(i), E(rootID, childID));
                  firstCollectionIteration = false;
              }
          }
      }
  }

  LOG4CXX_INFO(logger, std::setw(80) << std::setfill('-') << "-");
  LOG4CXX_INFO(logger, "MIN ERROR: " << minOutputValue << " at root "
               << minRootPos.x << " " << minRootPos.y);

  t_recursion = (double) cvGetTickCount() - t_recursion;
  t_all = (double) cvGetTickCount() - t_all;

  LOG4CXX_INFO(logger,
               "time for init maps: "
               << (t_allPolynoms_1 / ((double) cvGetTickFrequency() * 1000000.)));
  LOG4CXX_INFO(logger,
               "time recursion img "
               << (t_recursion / ((double) cvGetTickFrequency() * 1000000.))
               << " s");
  LOG4CXX_INFO(logger,
               "total answer time "
               << (t_all / ((double) cvGetTickFrequency() * 1000000.))
               << " s");

  rectangle(gui_, scaledRect, cv::Scalar(255, 0, 0));

  rectangle(gui_, Rect(midPoint.x - SURROUNDING_MIDPOINT_DETECTION_RADIUS,
                       midPoint.y - SURROUNDING_MIDPOINT_DETECTION_RADIUS,
                       2 * SURROUNDING_MIDPOINT_DETECTION_RADIUS,
                       2 * SURROUNDING_MIDPOINT_DETECTION_RADIUS),
            cv::Scalar(150, 150, 150));

  LOG4CXX_INFO(logger, std::setw(80) << std::setfill('-') << "-");


  localPolynoms_->DrawAnswerMap(12);


  first_detection_ = false;
}

void faceAnalysis::CPT::SetImage(const cv::Mat &image, const cv::Rect &detection)
{
  current_image_ = image;
  current_face_detection_ = detection;
  imageSet = true;
}

std::map<int, std::pair<int, int> > faceAnalysis::CPT::unscaled_collected_points() {
  std::map<int, std::pair<int, int> > unscaled_points;

  for(std::map<int,std::pair<int,int> >::const_iterator itr = collected_face_points_.begin(); itr != collected_face_points_.end(); ++itr)
    {
      unscaled_points[itr->first] = std::pair<int,int>(itr->second.first / current_scale_factor_,itr->second.second / current_scale_factor_);

    }

  return unscaled_points;
}

void faceAnalysis::CPT::FreePolynoms() {
  localPolynoms_->FreeAllPolynoms();
}

void faceAnalysis::CPT::UpdateGlobal(const cv::Point2i better_li,
                                     const cv::Point2i better_lj) {
  globalModel_->UpdateEdge(better_li, better_lj);
}


double faceAnalysis::CPT::AnswerChild(const int parent_pos_x, const int parent_pos_y,
                                      const TreeVertex current_vertex, const E current_edge)  {

  // get mean position of target -- w neighborhood
  // TODO include variance
  cv::Point2d relMeanPoint = globalModel_->mean_point(current_edge);
  globalModel_->set_current_edge(current_edge);


  int col_max = round(parent_pos_x - relMeanPoint.x + SEARCHWINDOW_RADIUS);
  int row_max = round(parent_pos_y - relMeanPoint.y + SEARCHWINDOW_RADIUS);
  int col_min = round(parent_pos_x - relMeanPoint.x - SEARCHWINDOW_RADIUS);
  int row_min = round(parent_pos_y - relMeanPoint.y - SEARCHWINDOW_RADIUS);

    cv::Point minPoint = cv::Point2i(0, 0);
  std::vector<TreeVertex> childs = globalModel_->childs(current_vertex);

  if (childs.size() == 0)
    {
      if (GetElementBc(parent_pos_x,
                       parent_pos_y, (int) current_vertex) == 0)
        {
          double minTotalValue = std::numeric_limits<double>::max();
          for (int col_itr = col_min; col_itr < col_max; col_itr++)
            {
              if (col_itr >= round(all_face_points_means_[(int) current_vertex].x) - SURROUNDING_TO_CHECK
                  && col_itr <= round(all_face_points_means_[(int) current_vertex].x) + SURROUNDING_TO_CHECK)
                {
                  for (int row_itr = row_min; row_itr < row_max; row_itr++) {
                      if (row_itr >= round(all_face_points_means_[(int) current_vertex].y) - SURROUNDING_TO_CHECK
                          && row_itr <= round(all_face_points_means_[(int) current_vertex].y) + SURROUNDING_TO_CHECK)
                        {
                          bool outside = false;
                          if (col_itr < 0 || row_itr < 0 || row_itr >= scaled_whole_image_.rows || col_itr >= scaled_whole_image_.cols)
                            {
                              outside = true;
                            }
                          if (!outside)
                            {
                              if (localPolynoms_->GetElementFeatureMap(row_itr, col_itr, 0) == 0)
                                {
                                  localPolynoms_->GenerateFeature(row_itr, col_itr);
                                }
                            }
                          if (!localPolynoms_->IsAnswerMapInitialized((int) current_edge.second))
                            {
                              localPolynoms_->initAnswer((int) current_edge.second);
                            }

                          double localAnswer;
                          if (!outside)
                            {
                              localAnswer = 1 -  localPolynoms_->getAnswer((int) current_edge.second,row_itr, col_itr);
                              if (localAnswer == 1)
                                {
                                  localAnswer =  1 - localPolynoms_->GenerateAnswerForPolynom((int) current_edge.second, row_itr, col_itr);
                                }
                            } else
                            {
                              localAnswer = 1 - NEGATIVE_VALUE;
                            }

                          double globalAnswer = globalModel_->CalcMahalanobis(parent_pos_x, parent_pos_y, col_itr, row_itr );

                          //                                                double globalAnswer = 1 - exp(-globalModel->calcMahalanobis(Point2i(parent_pos_x, parent_pos_y), Point2i(ty, tx), current_edge));

                          double totalValueTemp = (weight_local_ * localAnswer)
                              + (weight_global_ * globalAnswer);

                          if (totalValueTemp < minTotalValue)
                            {
                              minTotalValue = totalValueTemp;
                              minPoint.x = col_itr;
                              minPoint.y = row_itr;
                            }
                        }
                    }
                }
            }

          SetElementBc(parent_pos_x, parent_pos_y, (int) current_vertex, minTotalValue);
          SetElementBcPosition(parent_pos_x, parent_pos_y, (int) current_vertex, minPoint.x, minPoint.y);
          return minTotalValue;

        } else
        {
          return GetElementBc(parent_pos_x, parent_pos_y, (int) current_vertex);
        }

    } else
    {
      double minTotalValue = std::numeric_limits<double>::max();

      if (GetElementBc(parent_pos_x, parent_pos_y, (int) current_vertex) == 0)
        {
          for (int col_itr = col_min; col_itr < col_max; col_itr++) {
              if (col_itr >= round(all_face_points_means_[(int) current_vertex].x) - SURROUNDING_TO_CHECK
                  && col_itr <= round(all_face_points_means_[(int) current_vertex].x) + SURROUNDING_TO_CHECK)
                {
                  for (int row_itr = row_min; row_itr < row_max; row_itr++)
                    {
                      if (row_itr >= round(all_face_points_means_[(int) current_vertex].y) - SURROUNDING_TO_CHECK
                          && row_itr <= round(all_face_points_means_[(int) current_vertex].y) + SURROUNDING_TO_CHECK)
                        {
                          bool outside = false;
                          if ((col_itr < 0 || row_itr < 0 || row_itr >= scaled_whole_image_.rows || col_itr >= scaled_whole_image_.cols))
                            {
                              outside = true;
                            }

                          double localAnswer;
                          if (!outside)
                            {
                              if (localPolynoms_->GetElementFeatureMap(row_itr, col_itr, 0) == 0) {
                                  localPolynoms_->GenerateFeature(row_itr, col_itr);
                                }
                            }

                          if (!localPolynoms_->IsAnswerMapInitialized(current_edge.second))
                            {
                              localPolynoms_->initAnswer(current_edge.second);
                            }

                          if (!outside)
                            {
                              localAnswer =  1 - localPolynoms_->getAnswer(current_edge.second,row_itr, col_itr);
                              if (localAnswer == 1)
                                {
                                  localAnswer =  1 - localPolynoms_->GenerateAnswerForPolynom(current_edge.second, row_itr, col_itr);
                                }
                            } else
                            {
                              localAnswer = 1 - NEGATIVE_VALUE;
                            }


                          double globalAnswer = globalModel_->CalcMahalanobis(parent_pos_x, parent_pos_y, col_itr, row_itr);
                          //                        double globalAnswer = 1 - exp(-globalModel->calcMahalanobis(Point2i(parent_pos_x, parent_pos_y), Point2i(ty, tx), current_edge));

                          double totalValueTemp = weight_local_ * localAnswer
                              + (weight_global_ * globalAnswer);

                          for (uint i = 0; i < childs.size(); i++)
                            {
                              int childID = (int) childs.at(i);

                              if (GetElementBc(col_itr, row_itr, childID) == 0)
                                {
                                  double childAnswer =
                                      AnswerChild(col_itr, row_itr, childs.at(i), E(current_edge.second, childID));
                                  totalValueTemp += childAnswer;
                                } else
                                {
                                  double childAnswer = GetElementBc(col_itr, row_itr , childID);
                                  totalValueTemp += childAnswer;
                                }
                            }

                          if (totalValueTemp < minTotalValue)
                            {
                              minTotalValue = totalValueTemp;
                              minPoint.x = col_itr;
                              minPoint.y = row_itr;
                            }
                        }
                    }
                }
            }

          SetElementBc(parent_pos_x, parent_pos_y, (int) current_vertex, minTotalValue);
          SetElementBcPosition(parent_pos_x, parent_pos_y, (int) current_vertex, minPoint.x, minPoint.y);
          return minTotalValue;

        } else
        {
          return GetElementBc(parent_pos_x, parent_pos_y, (int) current_vertex);
        }
    }

  // TODO if point not in image --> discard
}

void faceAnalysis::CPT::UpdateLocal(const cv::Point2i position,
                                    const int polynom_id) {
  LOG4CXX_INFO(logger, "update local " << polynom_id << " " << position);

  Gauss gauss;
  gauss.initGauss((GAUSS_RADIUS_UPDATE * 2) + 1);

  //    for (int neigIndexX = -GAUSS_RADIUS_UPDATE; neigIndexX <= GAUSS_RADIUS_UPDATE; neigIndexX++) {
  //        for (int neigIndexY = -GAUSS_RADIUS_UPDATE; neigIndexY <= GAUSS_RADIUS_UPDATE; neigIndexY++) {
  int xSur = position.x;
  int ySur = position.y;

  if (localPolynoms_->GetElementFeatureMap(xSur, ySur, 0) == 0)
    {
      localPolynoms_->SetCurrentImage(scaled_whole_image_);
      localPolynoms_->GenerateFeature(xSur, ySur);
    }

  localPolynoms_->GenerateAnswerForPolynom(polynom_id, ySur, xSur);
  //            double val = 1 - gauss.gauss2D(neigIndexX, neigIndexY);

  localPolynoms_->Update(polynom_id, xSur, ySur, POSITIVE_VALUE, 0.008);
  //        }
  //    }
  //    LOG4CXX_INFO(logger, "update " << polynomID << " " << position);
}

void faceAnalysis::CPT::CollectPositions(bool recursion_first_time, int parent_pos_x,
                                         int parent_pos_y,
                                         TreeVertex current_vertex,
                                         E current_edge) {
  std::vector<TreeVertex> childs = globalModel_->childs(current_vertex);
  std::pair<int, int> point = GetElementBcPosition(parent_pos_x, parent_pos_y, (int) current_vertex);
  double value = GetElementBc(parent_pos_x, parent_pos_y, (int) current_vertex);

  collected_face_points_[(int) current_vertex] = point;
  collected_face_points_values_[(int) current_vertex] = value;

  if (childs.size() != 0)
    {
      for (uint i = 0; i < childs.size(); i++)
        {
          int childID = (int) childs.at(i);
          CollectPositions(recursion_first_time, point.first, point.second, childs.at(i),
                           E(current_edge.second, childID));
        }
    }
}

void faceAnalysis::CPT::GenerateB_c(const int image_cols,
                                    const int image_rows) {
  b_c_size_x = image_cols;
  b_c_size_y_ = image_rows;

  //TODO realcv::Size
  b_c_size_z_ = 100;
  //    B_c_z = globalModel->getcv::Size();

  b_c_ = new double[b_c_size_x * b_c_size_y_ * b_c_size_z_];
  b_c_position_ = new int[b_c_size_x * b_c_size_y_ * b_c_size_z_ * 2];
}


void faceAnalysis::CPT::FreeBc() {
  delete [] b_c_;
}

void faceAnalysis::CPT::FreeBcPosition() {
  delete [] b_c_position_;
}

void faceAnalysis::CPT::SetElementBc(const int idx, const int jdx,
                                     const int kdx, const double value) {
  b_c_[kdx * b_c_size_x * b_c_size_y_ + idx * b_c_size_y_ + jdx ] = value;
  //    B_c[idx * B_c_y * B_c_z + jdx * B_c_z + kdx ] = value;
}

double faceAnalysis::CPT::GetElementBc(int idx, int jdx, int kdx) const {
  return b_c_[kdx * b_c_size_x * b_c_size_y_ + idx * b_c_size_y_ + jdx ];
}

void faceAnalysis::CPT::SetElementBcPosition(const int idx, const int jdx,
                                             const int kdx, const int point_x,
                                             const int point_y)
{
  b_c_position_[kdx * b_c_size_x * b_c_size_y_ * 2 + idx * b_c_size_y_ * 2 + jdx * 2 ] = point_x;
  b_c_position_[kdx * b_c_size_x * b_c_size_y_ * 2 + idx * b_c_size_y_ * 2 + jdx * 2 + 1] = point_y;
}

std::pair<int, int> faceAnalysis::CPT::GetElementBcPosition(int idx,
                                                            int jdx,
                                                            int kdx) const {
  std::pair<int, int> retVal = std::pair<int, int>(
        b_c_position_[kdx * b_c_size_x * b_c_size_y_ * 2 + idx * b_c_size_y_ * 2 + jdx * 2],
      b_c_position_[kdx * b_c_size_x * b_c_size_y_ * 2 + idx * b_c_size_y_ * 2 + jdx * 2 + 1]);

  return retVal;
}
