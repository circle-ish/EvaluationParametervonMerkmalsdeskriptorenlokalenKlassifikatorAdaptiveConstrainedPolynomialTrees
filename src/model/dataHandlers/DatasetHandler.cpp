#include "model/dataHandlers/DatasetHandler.h"
#include "model/local/GlobalDefinitions.h"
#include <fstream>
#include <opencv2/highgui/highgui.hpp>

#include "model/detectors/ViolaLienhartDetector.h"
#include "model/GlobalConstants.h"


log4cxx::LoggerPtr faceAnalysis::DatasetHandler::logger(
    log4cxx::Logger::getLogger("CPT.models.training.trainDataHandler"));

faceAnalysis::DatasetHandler::DatasetHandler(std::string cascade)
  : cascade_(cascade) {
  srand(time(0));
  detector_ = DetectorPtr(new ViolaLienhartDetector(cascade));
}

faceAnalysis::DatasetHandler::DatasetHandler() {
  srand(time(0));
}

std::vector<std::string> faceAnalysis::DatasetHandler::GenerateSamples(
    const int number, const std::vector<std::string> list_of_all_files) {

  std::vector<int> random_numbers;
  std::vector<std::string> used_files;

  int j = 0;
  int i = 0;
  while (j < number)
    {
      i++;
      int number = rand() % list_of_all_files.size() + 1 - 1;

      if (find(random_numbers.begin(), random_numbers.end(), number)
          == random_numbers.end())
        {
          random_numbers.push_back(number);
          used_files.push_back(list_of_all_files.at(number));
          j++;
        };
    }

  return used_files;
}

void faceAnalysis::DatasetHandler::WriteUsedFileNames(
    const std::string name, const std::vector<std::string> list_of_used_files) {

  std::ofstream output_stream;
  output_stream.open(name.c_str());
  for (std::vector<std::string>::const_iterator it = list_of_used_files.begin();
       it != list_of_used_files.end(); ++it)
    {
      output_stream << it->c_str() << "\n";
    }
  output_stream.close();
}

std::vector<std::string> faceAnalysis::DatasetHandler::ReadFileNames(
    const std::string path_to_file_containing_names) {
  std::vector<std::string> file_list;
  std::string line;
  std::ifstream file(path_to_file_containing_names.c_str());
  int count = 222;
  if (file.is_open())
    {
      while (file.good() && count > 0)
        {
          getline(file, line);
          if (line.compare("") != 0)
            {
              --count;
              file_list.push_back(line);
            }
        }
      file.close();
    } else
    {
      LOG4CXX_ERROR(logger, "Cannot open file " << path_to_file_containing_names
                    << "-- EXIT");
      exit(-1);
    }

  return file_list;
}

Dataset faceAnalysis::DatasetHandler::ReadDataset(
    const std::vector<std::string> readed_data_names,
    const std::string path_to_point_folder,
    const std::string path_to_image_folder, const DataBaseType database_type) {

  Dataset dataset;

  LOG4CXX_INFO(logger, "Read " << readed_data_names.size() << " images");

  switch (database_type) {
    case MULTIPIE:
      dataset = ReadMultipieDataset(readed_data_names, path_to_point_folder,
                                 path_to_image_folder);
      break;
    case LFPW_TEST_300W:
      dataset = ReadOtherDataset(readed_data_names, path_to_point_folder,
                              path_to_image_folder, ".png");
      break;
    case LFPW_TRAIN_300W:
      dataset = ReadOtherDataset(readed_data_names, path_to_point_folder,
                              path_to_image_folder, ".png");
      break;
    default:
      dataset = ReadOtherDataset(readed_data_names, path_to_point_folder,
                              path_to_image_folder, ".jpg");
      break;
    }

  return dataset;
}

std::vector<cv::Rect> faceAnalysis::DatasetHandler::ReadBoundingBoxes(
    const std::vector<std::string> readed_data_names,
    const std::string path_to_bounding_boxes,
    const bool using_ground_truth) {

  std::vector<cv::Rect> readed_bounding_boxes;

  if (!path_to_bounding_boxes.empty())
    {
      int bounding_box_counter = 0;
      for (std::vector<std::string>::const_iterator itr =
           readed_data_names.begin(); itr != readed_data_names.end(); ++itr)
        {
          bounding_box_counter++;
          if (verbosity > 3) {
              LOG4CXX_INFO(logger, "Read bounding box " << bounding_box_counter
                       << "/" << readed_data_names.size());
          }

          std::stringstream fullPtrPath;
          fullPtrPath << path_to_bounding_boxes << "/" << *itr;

          cv::FileStorage fs(fullPtrPath.str(), cv::FileStorage::READ);

          cv::Mat boundingBox;
          if (using_ground_truth)
            fs["groundTruth"] >> boundingBox;
          else
            fs["detector"] >> boundingBox;

          fs.release();

          cv::Rect boundingBoxRect = cv::Rect(
                boundingBox.at<double>(0), boundingBox.at<double>(1),
                boundingBox.at<double>(2) - boundingBox.at<double>(0),
                boundingBox.at<double>(3) - boundingBox.at<double>(1));

          readed_bounding_boxes.push_back(boundingBoxRect);
        }
    } else
    {
      LOG4CXX_WARN(logger, "no bounding boxes defined");
    }

  return readed_bounding_boxes;
}

std::vector<cv::Rect> faceAnalysis::DatasetHandler::ExtractFaceBoxes(
    Dataset evaluation_data) {

  std::vector<cv::Rect> extracted_face_boxes;
  int extracted_bounding_box_counter = 0;

  for (Dataset::const_iterator itr = evaluation_data.begin();
       itr != evaluation_data.end(); ++itr)
    {
      extracted_bounding_box_counter++;
      LOG4CXX_DEBUG(logger, "Generate bounding box "
                    << extracted_bounding_box_counter << "/"
                    << evaluation_data.size());

      DatasetExample sample = *itr;
      PointsOfImage points = sample.second;

      double biggest_x = std::numeric_limits<double>::min();
      double biggest_y = std::numeric_limits<double>::min();
      double smallest_x = std::numeric_limits<double>::max();
      double smallest_y = std::numeric_limits<double>::max();

      for (int annotated_point_counter = 0;
           annotated_point_counter < points.rows; annotated_point_counter++)
        {
          double x = points.at<float>(annotated_point_counter, 0);
          double y = points.at<float>(annotated_point_counter, 1);

          if (biggest_x < x)
            biggest_x = x;

          if (biggest_y < y)
            biggest_y = y;

          if (smallest_x > x)
            smallest_x = x;

          if (smallest_y > y)
            smallest_y = y;
        }

      extracted_face_boxes.push_back(cv::Rect(smallest_x, smallest_y,
                                              biggest_x - smallest_x,
                                              biggest_y - smallest_y));
    }

  return extracted_face_boxes;
}

Dataset faceAnalysis::DatasetHandler::TransformTraining(
    const Dataset training_data, const std::string path_to_bounding_boxes,
    const std::vector<std::string> list_of_all_files) {

  Dataset dataset;

  std::vector<cv::Rect> bounding_boxes = ReadBoundingBoxes(
        list_of_all_files, path_to_bounding_boxes, false);

  for (unsigned int scale_counter = 0; scale_counter < training_data.size();
       scale_counter++)
    {
      if (verbosity > 3) {
          LOG4CXX_INFO(logger, "Scale and detect face: " << scale_counter << "/"
                   << training_data.size());
      }

      DatasetExample sample = training_data.at(scale_counter);
      cv::Mat grayscale_img;
      cvtColor(sample.first, grayscale_img, CV_BGR2GRAY);

      cv::Rect face_box;
      if (path_to_bounding_boxes.empty())
        {
          face_box = ExtractFaceBox(sample);
        } else
        {
          if (verbosity > 3) {
              LOG4CXX_INFO(logger, "Read bounding box");
          }
          face_box = bounding_boxes.at(scale_counter);
        }

      double scale_factor = NORM_FACE_WIDTH / face_box.width;

      cv::Mat scaled_image;
      resize(grayscale_img, scaled_image,
             cv::Size(scale_factor * grayscale_img.cols,
                      scale_factor * grayscale_img.rows));

      sample.first = scaled_image;

      for (int point_counter = 0; point_counter < sample.second.rows;
           point_counter++)
        {
          sample.second.at<float>(point_counter, 0) =
              scale_factor * sample.second.at<float>(point_counter, 0);
          sample.second.at<float>(point_counter, 1) =
              scale_factor * sample.second.at<float>(point_counter, 1);
        }

      dataset.push_back(sample);
    }

  return dataset;
}

cv::Rect faceAnalysis::DatasetHandler::ExtractFaceBox(
    const DatasetExample training_example) {

  PointsOfImage points = training_example.second;

  double biggest_x = std::numeric_limits<double>::min();
  double biggest_y = std::numeric_limits<double>::min();
  double smallest_x = std::numeric_limits<double>::max();
  double smallest_y = std::numeric_limits<double>::max();

  for (int annotated_point_count = 0; annotated_point_count < points.rows;
       annotated_point_count++)
    {
      double x = points.at<float>(annotated_point_count, 0);
      double y = points.at<float>(annotated_point_count, 1);

      if (biggest_x < x)
        biggest_x = x;

      if (biggest_y < y)
        biggest_y = y;

      if (smallest_x > x)
        smallest_x = x;

      if (smallest_y > y)
        smallest_y = y;
    }

  return cv::Rect(smallest_x, smallest_y, biggest_x - smallest_x,
                  biggest_y - smallest_y);
}

cv::Mat faceAnalysis::DatasetHandler::GrayScaleEnergyNormalization(
    const cv::Mat image_part) {
  cv::Mat normed_image(image_part.size(), image_part.type());

  double sum_of_all_pixels = 0;
  for (int i = 0; i < image_part.rows; i++)
    {
      for (int j = 0; j < image_part.cols; j++)
        {
          sum_of_all_pixels += image_part.at<double>(i, j);
        }
    }

  if (sum_of_all_pixels != 0)
    {
      for (int i = 0; i < image_part.rows; i++)
        {
          for (int j = 0; j < image_part.cols; j++) {
              normed_image.at<double>(i, j) =
                  (double) image_part.at<double>(i, j)
                  * (ENERGY_NORM_VALUE / sum_of_all_pixels);
            }
        }
    } else
    {
      LOG4CXX_INFO(logger, "Cannot norm energy of part -- all pixels = 0");
    }

  return normed_image;
}


Dataset faceAnalysis::DatasetHandler::ReadMultipieDataset(
    const std::vector<std::string> readed_data_names,
    const std::string path_to_point_folder,
    const std::string path_to_image_folder) {

  int image_counter = 0;
  Dataset readed_data;

  for (std::vector<std::string>::const_iterator itr = readed_data_names.begin();
       itr != readed_data_names.end(); ++itr)
    {
      image_counter++;
      LOG4CXX_INFO(logger, "Read image " << image_counter << "/"
                   <<  readed_data_names.size());

      std::string file_path = *itr;
      std::string based_ptr_name = file_path.substr(
            path_to_point_folder.length(), 16);
      std::string subject_id = based_ptr_name.substr(0, 3);
      std::string session_number = based_ptr_name.substr(4, 2);
      std::string recording_number = based_ptr_name.substr(7, 2);
      std::string camera_label = based_ptr_name.substr(10, 3);
      std::string img_number = based_ptr_name.substr(14, 2);
      UNUSED(img_number);

      // only frontal faces
      if (!camera_label.compare("051"))
        {
          PointsOfImage annotated_points;
          cv::FileStorage file(file_path, cv::FileStorage::READ);
          file["pts"] >> annotated_points;
          file.release();

          std::stringstream image_path;
          image_path << path_to_image_folder << "/session" << session_number
                     << "/multiview/" << subject_id << "/" << recording_number
                     << "/" << camera_label.substr(0, 2) << "_"
                     << camera_label.substr(2, 1) << "/" << based_ptr_name
                     << ".png";

          cv::Mat image = cv::imread(image_path.str());

          //          // HACK TO REMOVE POINT 60 AND 64 // Achtung, hier ab 0 auf Bilder zählen ab 1
          //          if ((!(annotated_points.rows == 66)))
          //            {
          //              int newCounter = 0;
          //              PointsOfImage ptsNew(annotated_points.rows - 2, annotated_points.cols, annotated_points.type());
          //              for (int i = 0; i < annotated_points.rows; i++) {
          //                  if ((i == 60) || (i == 64)) {
          //                      //                             LOG4CXX_DEBUG(logger, "SKIP row " << i );
          //                    } else {
          //                      ptsNew.at<float>(newCounter, 0) = annotated_points.at<float>(i, 0);
          //                      ptsNew.at<float>(newCounter, 1) = annotated_points.at<float>(i, 1);

          //                      //                            circle(image, Point(ptsNew.at<float>(newCounter, 0), ptsNew.at<float>(newCounter, 1)), 2, Scalar(255, 0, 0));
          //                      //                            imshow("scaledImg", image);
          //                      //                            waitKey(1);
          //                      //                    LOG4CXX_DEBUG(logger, ptsNew.at<float>(newCounter, 0) << " " << ptsNew.at<float>(newCounter, 1) );

          //                      newCounter++;
          //                    }
          //                }

          //              DatasetExample onePair;
          //              onePair = std::make_pair(image, ptsNew);
          //              readed_data.push_back(onePair);
          //            } else
          //            {
          //                     LOG4CXX_DEBUG(logger, "Skipped reindexing" );

          DatasetExample onePair;
          onePair = std::make_pair(image, annotated_points);
          readed_data.push_back(onePair);

          //            }
        }
    }
  return readed_data;
}

Dataset faceAnalysis::DatasetHandler::ReadOtherDataset(
    const std::vector<std::string> readed_data_names,
    const std::string path_to_point_folder,
    const std::string path_to_image_folder, const std::string suffix) {

  int image_counter = 0;
  Dataset readed_data;

  for (std::vector<std::string>::const_iterator itr =
       readed_data_names.begin(); itr != readed_data_names.end(); ++itr)
    {
      image_counter++;
      if (verbosity > 3) {
          LOG4CXX_INFO(logger, "Read image " << image_counter << "/"
                   << readed_data_names.size());
      }
      std::stringstream full_path_to_points;
      full_path_to_points << path_to_point_folder << "/" << *itr;

      PointsOfImage points;
      cv::FileStorage point_file(full_path_to_points.str(),
                                 cv::FileStorage::READ);
      point_file["pts"] >> points;
      point_file.release();

      std::string based_points_name =
          full_path_to_points.str().substr(path_to_point_folder.length());
      based_points_name.erase(based_points_name.length() - 4);

      std::stringstream path_to_image;
      path_to_image << path_to_image_folder << based_points_name << suffix;

      cv::Mat image = cv::imread(path_to_image.str());

      DatasetExample one_pair;
      one_pair = std::make_pair(image, points);
      readed_data.push_back(one_pair);
    }

  return readed_data;
}
