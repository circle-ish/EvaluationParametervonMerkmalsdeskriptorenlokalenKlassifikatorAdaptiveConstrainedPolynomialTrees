#include "TotalEvaluator.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

#include "model/global/GlobalTree.h"
#include "model/trackers/CPT.h"

log4cxx::LoggerPtr faceAnalysis::TotalEvaluator::logger(
    log4cxx::Logger::getLogger("CPT.models.evaluation.totalEvaluator"));


faceAnalysis::TotalEvaluator::TotalEvaluator() {

}

faceAnalysis::TotalEvaluator::~TotalEvaluator() {

}

double faceAnalysis::TotalEvaluator::Evaluate(const std::vector<int> landmarks_to_evaluate) {
  UNUSED(landmarks_to_evaluate);

  CPTPtr cpt = faceAnalysis::CPTPtr(new faceAnalysis::CPT(configuration_,
                                                          std::vector<int>()));
  cpt->Load(path_to_trained_model_);

  int counter = 0;
  int skipped = 0;
  double total_error = 0;

  std::map<int, std::pair<int, int> > positions;

  // TODO variable
  std::string path_to_csv = configuration_.get<std::string>(
        "evaluation.pathToOutputCSV");

  std::ofstream output_evaluation_file(path_to_csv.c_str());

  for (Dataset::const_iterator itr = dataset_.begin(); itr != dataset_.end();
       ++itr)
    {
      LOG4CXX_INFO(logger, "Image " << counter << "/" << dataset_.size());

      DatasetExample ground_truth = *itr;
      cv::cvtColor(ground_truth.first, ground_truth.first, CV_RGB2GRAY);

      cv::Rect detector_output = bounding_boxes_.at(counter);

      double scale_factor = NORM_FACE_WIDTH / (double) detector_output.width;

      cv::Rect scaled_face_detection;
      scaled_face_detection.x = round(scale_factor * detector_output.x);
      scaled_face_detection.y = round(scale_factor * detector_output.y);
      scaled_face_detection.width = round(scale_factor * detector_output.width);
      scaled_face_detection.height =
          round(scale_factor * detector_output.height);

      cv::Mat scaled_image;
      resize(ground_truth.first, scaled_image,
             cv::Size(scale_factor * ground_truth.first.cols,
                      scale_factor * ground_truth.first.rows));

      if (scaled_face_detection.tl().x < 0)
        scaled_face_detection.x = 0;
      if (scaled_face_detection.tl().y < 0)
        scaled_face_detection.y = 0;
      if (scaled_face_detection.br().x > scaled_image.cols)
        scaled_face_detection.width = scaled_image.cols;
      if (scaled_face_detection.br().y > scaled_image.rows)
        scaled_face_detection.height = scaled_image.cols;

      cv::Point2f root_position_unscaled =
          cv::Point2f(ground_truth.second.at<float>(cpt->root_ID(), 0),
                      ground_truth.second.at<float>(cpt->root_ID(), 1));

      cpt->SetImage(ground_truth.first, detector_output);
      cpt->DetectFeatures(true, root_position_unscaled);

      positions = cpt->unscaled_collected_points();

      // Create a vector containing the channels of the new colored image
      std::vector<cv::Mat> channels;

      channels.push_back(ground_truth.first); // 1st channel
      channels.push_back(ground_truth.first); // 2nd channel
      channels.push_back(ground_truth.first); // 3rd channel

      // Construct a new 3-channel image of the same size and depth
      cv::Mat gui;
      cv::merge(channels, gui);

      // TODO stimmt das auch bei franck dataset?!
      cv::Point2d ground_truth_left_eye(ground_truth.second.at<float>(27, 0),
                                        ground_truth.second.at<float>(27, 1));
      cv::Point2d ground_truth_right_eye(ground_truth.second.at<float>(32, 0),
                                         ground_truth.second.at<float>(32, 1));

      double evaluation_norm = norm(ground_truth_left_eye
                                    - ground_truth_right_eye);
      double sum_error = 0;

      for (std::map<int, std::pair<int, int> >::const_iterator itr = positions.begin();
           itr != positions.end(); ++itr)
        {
          int ground_truth_id = -1;
          if (database_to_use_ == FRANCK)
            ground_truth_id = MapToFranckDataset(itr->first);
          else
            ground_truth_id = itr->first;

          cv::Point2d fitted_point(itr->second.first, itr->second.second);

          if (ground_truth_id != -1)
            {
              cv::Point2d ground_truth_point(
                    ground_truth.second.at<float>(ground_truth_id, 0),
                    ground_truth.second.at<float>(ground_truth_id, 1));

              double normed_value = norm(ground_truth_point - fitted_point);
              sum_error += (normed_value / (double) evaluation_norm);

              line(gui, ground_truth_point, fitted_point,
                   cv::Scalar(255, 255, 255), 2);

              circle(gui, ground_truth_point, 2, cv::Scalar(0, 255, 0), 2);
            }


          circle(gui, fitted_point, 2, cv::Scalar(0, 0, 255), 2);
        }

      LOG4CXX_INFO(logger, "Total error for this image " << sum_error);
      LOG4CXX_INFO(logger, "per point error for this image " << sum_error / 33);

      output_evaluation_file << sum_error << ";" << std::endl;

      while ((gui.cols > 1000) || (gui.rows > 1000))
        {
          resize(gui, gui, cv::Size(gui.cols / 2, gui.rows / 2));
        }

      imshow("RESULT", gui);
      cv::waitKey(0);

      std::stringstream output_path;
      output_path.clear();
      output_path << "image_all_0.08/" << counter << ".jpg";
      imwrite(output_path.str(), gui);

      LOG4CXX_INFO(logger, "Writing image to " << output_path.str());

      counter++;
    }

  double average_error = total_error / counter;

  LOG4CXX_INFO(logger, "-------------------------------------");
  LOG4CXX_INFO(logger, "Skipped " << skipped << " images");
  LOG4CXX_INFO(logger, "Avrg error for test set: " << average_error);
  LOG4CXX_INFO(logger, "-------------------------------------");
  return average_error;

}

int faceAnalysis::TotalEvaluator::MapToFranckDataset(const int point_id)
{
  switch (point_id) {
    case 0:
      return 0;
      break;
    case 8:
      return 7;
      break;
    case 16:
      return 14;
      break;
    case 17:
      return 21;
      break;
    case 21:
      return 24;
      break;
    case 22:
      return 18;
      break;
    case 26:
      return 15;
      break;
    case 30:
      return 67;
      break;
    case 31:
      return 40;
      break;
    case 33:
      return 41;
      break;
    case 35:
      return 42;
      break;
    case 36:
      return 27;
      break;
    case 39:
      return 29;
      break;
    case 42:
      return 34;
      break;
    case 45:
      return 32;
      break;
    case 48:
      return 48;
      break;
    case 49:
      return 49;
      break;
    case 50:
      return 50;
      break;
    case 51:
      return 51;
      break;
    case 52:
      return 52;
      break;
    case 53:
      return 53;
      break;
    case 54:
      return 54;
      break;
    case 55:
      return 55;
      break;
    case 56:
      return 56;
      break;
    case 57:
      return 57;
      break;
    case 58:
      return 58;
      break;
    case 59:
      return 59;
      break;
    case 61:
      return 65;
      break;
    case 62:
      return 64;
      break;
    case 63:
      return 63;
      break;
    case 65:
      return 62;
      break;
    case 66:
      return 61;
      break;
    case 67:
      return 60;
      break;
    default:
      return -1;
      break;
    }
}
