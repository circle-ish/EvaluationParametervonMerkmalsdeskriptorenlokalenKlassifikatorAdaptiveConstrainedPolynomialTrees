#include "DaimlerDataHandler.h"

#include <fstream>

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "model/detectors/ViolaLienhartDetector.h"

log4cxx::LoggerPtr faceAnalysis::DaimlerDataHandler::logger(
    log4cxx::Logger::getLogger("CPT.models.evaluation.daimlerDataHandler"));

faceAnalysis::DaimlerDataHandler::DaimlerDataHandler(
    const std::string folder_path, const std::string camera,
    const ClassifierType classifier,const std::string clm_data_path)
  : folder_path_(folder_path),
    camera_(camera),
    classifier_(classifier),
    clm_data_path_(clm_data_path) {}

std::vector<bool> faceAnalysis::DaimlerDataHandler::GenerateFrameDropList() {
  std::stringstream frame_input_name;
  frame_input_name << folder_path_ << "/csv_frame_" << camera_ << ".csv";
  std::ifstream frame_input_file(frame_input_name.str().c_str());

  std::vector<bool> frame_drop;
  std::string line;
  if (frame_input_file.is_open())
    {
      int dropLineCounter = 0;
      while (getline(frame_input_file, line))
        {
          if (dropLineCounter == 2)
            {
              std::string delimiter = ";";
              size_t pos = 0;
              std::string token;
              while ((pos = line.find(delimiter)) != std::string::npos)
                {
                  token = line.substr(0, pos);
                  line.erase(0, pos + delimiter.length());

                  if (!token.compare(""))
                    {
                      frame_drop.push_back(false);
                    }

                  if (!token.compare("1"))
                    {
                      frame_drop.push_back(true);
                    }
                }
              break;
            }
          dropLineCounter++;
        }
      frame_input_file.close();
    }

  return frame_drop;
}

cv::Mat faceAnalysis::DaimlerDataHandler::ConvertImage(const cv::Mat raw_image) {

  short max_value = std::numeric_limits<short>::min();
  short min_value = std::numeric_limits<short>::max();

  for (int i = 0; i < raw_image.rows; i++)
    {
      for (int j = 0; j < raw_image.cols; j++)
        {
          short val = raw_image.at<short>(i, j);
          if (val > max_value)
            max_value = val;

          if (val < min_value)
            min_value = val;
        }
    }

  short scale = max_value - min_value;
  cv::Mat converted_image(raw_image.rows, raw_image.cols, CV_8UC1);

  for (int i = 0; i < raw_image.rows; i++)
    {
      for (int j = 0; j < raw_image.cols; j++)
        {
          converted_image.at<uchar>(i, j) =
              ((double) raw_image.at<short>(i, j) / scale)*255;
        }
    }

  return converted_image;
}

std::vector<std::string> faceAnalysis::DaimlerDataHandler::GenerateCsv(
    const std::string cascade, const bool wait) {

  std::vector<bool> frame_drop = GenerateFrameDropList();
  std::vector<std::string> points;

  // TODO for all points generate init of file:
  //    for (int i = 0; i < trackerController->get2DShape().rows / 2; i++) {
  //        stringstream x_str;
  //        x_str << "landmark_" << i << "_x<all_num>;";
  //        points.push_back(x_str.str());
  //        stringstream y_str;
  //        y_str << "landmark_" << i << "_y<all_num>;";
  //        points.push_back(y_str.str());
  //    }

  std::stringstream head_pos_x;
  head_pos_x << "head_pos_x<all_num>;";
  std::stringstream head_pos_y;
  head_pos_y << "head_pos_y<all_num>;";
  std::stringstream head_pos_z;
  head_pos_z << "head_pos_z<all_num>;";
  std::stringstream head_px_x;
  head_px_x << "head_px_x<all_num>;";
  std::stringstream head_px_y;
  head_px_y << "head_px_y<all_num>;";
  std::stringstream head_conf;
  head_conf << "head_conf<all_num>;";
  std::stringstream head_ang_x;
  head_ang_x << "head_ang_x<all_num>;";
  std::stringstream head_ang_y;
  head_ang_y << "head_ang_y<all_num>;";
  std::stringstream head_ang_z;
  head_ang_z << "head_ang_z<all_num>;";
  std::stringstream frame_drop_id;
  frame_drop_id << "frame_drop_id<all_num>;";

  std::vector<std::string> path_vector;
  std::stringstream image_folder_stream;
  image_folder_stream << folder_path_ << "/" << camera_;

  for (boost::filesystem::recursive_directory_iterator end,
       dir(image_folder_stream.str()); dir != end; ++dir)
    {
      std::string pathString = dir->path().string();
      if (strstr(pathString.c_str(), ".tif") != NULL)
        {
          path_vector.push_back(pathString);
        }
    }

  sort(path_vector.begin(), path_vector.end());

  double path_vector_counter = 0;
  double counter = 0;
  DetectorPtr detector = DetectorPtr(new ViolaLienhartDetector(cascade));

  for (uint a = 0; a < frame_drop.size(); a++)
    {
      LOG4CXX_INFO(logger, "Counter " << counter << "/" << frame_drop.size());
      if (!frame_drop.at(counter))
        {
          frame_drop_id << ";";
          cv::Mat raw_image = cv::imread(path_vector.at(path_vector_counter),
                                         -1);

          path_vector_counter++;
          cv::Mat converted_frame= ConvertImage(raw_image);
          cv::Rect detector_rectangle;
          cv::Mat detector_image;

          std::map<int, cv::Rect> detection = detector->Detect(converted_frame);

          try {

            detector_rectangle = detection[0];
            detector_rectangle.height = detector_rectangle.height + 80;
            detector_rectangle.y = detector_rectangle.y - 40;
            //                    LOG4CXX_INFO(logger, detectorOutput);
            rectangle(converted_frame, detector_rectangle,
                      cv::Scalar(255, 255, 255));

            detector_image = converted_frame(detector_rectangle).clone();


            // TODO reset tracking
            //                trackerController->resetTracking();
            //                trackerController->setFailedBefore(true);

            // TODO track here
            //                trackerController->track(detectorImg);

            if (wait)
              {
                // TODO draw shape
                //                    trackerController->drawShape(imageFrameConv, detectorOutput);
                imshow("Image", converted_frame);
                cv::waitKey(0);
              }

            for (uint i = 0; i < points.size() / 2; i++)
              {
                // TODO get point to write to csv
                //                    cv::Point2d point = trackerController->get2DShapePoint(i);
                //
                //                    stringstream x_str;
                //                    x_str << points.at(i * 2) << point.x + detectorOutput.x << ";";
                //                    stringstream y_str;
                //                    y_str << points.at(i * 2 + 1) << point.y + detectorOutput.y << ";";
                //
                //                    points.at(i * 2) = x_str.str();
                //                    points.at(i * 2 + 1) = y_str.str();
              }

          } catch (...) {
            LOG4CXX_INFO(logger, "Error!");
            for (uint i = 0; i < points.size() / 2; i++) {
                std::stringstream x_str;
                x_str << points.at(i * 2) << ";";
                std::stringstream y_str;
                y_str << points.at(i * 2 + 1) << ";";

                points.at(i * 2) = x_str.str();
                points.at(i * 2 + 1) = y_str.str();
              }
          }

        } else
        {
          LOG4CXX_INFO(logger, "Framedrop!");
          frame_drop_id << "1;";
          for (uint i = 0; i < points.size() / 2; i++) {
              std::stringstream x_str;
              x_str << points.at(i * 2) << ";";
              std::stringstream y_str;
              y_str << points.at(i * 2 + 1) << ";";

              points.at(i * 2) = x_str.str();
              points.at(i * 2 + 1) = y_str.str();

            }
        }

      head_pos_x << ";";
      head_pos_y << ";";
      head_pos_z << ";";
      head_px_x << ";";
      head_px_y << ";";
      head_conf << ";";
      head_ang_x << ";";
      head_ang_y << ";";
      head_ang_z << ";";

      counter++;
    }

  points.push_back(head_pos_x.str());
  points.push_back(head_pos_y.str());
  points.push_back(head_pos_z.str());
  points.push_back(head_px_x.str());
  points.push_back(head_px_y.str());
  points.push_back(head_conf.str());
  points.push_back(head_ang_x.str());
  points.push_back(head_ang_y.str());
  points.push_back(head_ang_z.str());
  points.push_back(frame_drop_id.str());

  return points;
}

void faceAnalysis::DaimlerDataHandler::SaveFile(
    const std::vector<std::string> points) {

  std::stringstream output_path_stream;
  output_path_stream << folder_path_ << "/csv_clm_";

  switch (classifier_)
    {
    case POLY_FULL_QUADRATIC:
      output_path_stream << "polynom_full_quadratic_";
      break;
    case POLY_ONLY_QUADRATIC_LINEAR:
      output_path_stream << "polynom_quadratic_linear_";
      break;
    case POLY_LINEAR:
      output_path_stream << "polynom_linear_";
      break;
    default:
      break;
    }

  std::ofstream output_file;
  output_path_stream << camera_ << ".csv";
  output_file.open(output_path_stream.str().c_str());
  output_file << "first_frame_nbr<first_num>;1" << std::endl;

  for (uint i = 0; i < points.size(); i++)
    {
      output_file << points.at(i) << std::endl;
    }

  output_file.close();
}


// Anwendung:
//void CPT::handleDaimlerData(
//    std::string daimlerFolderPath,
//    std::string camera,
//    std::string clmDataPath,
//    bool waitIsSet
//    ) {
//  DaimlerDataHandler dataHandler(
//        daimlerFolderPath,
//        camera,
//        classifier,
//        clmDataPath
//        );
//  vector<string> points = dataHandler.generateCsv(cascade, waitIsSet);
//  dataHandler.saveFile(points);
//}
