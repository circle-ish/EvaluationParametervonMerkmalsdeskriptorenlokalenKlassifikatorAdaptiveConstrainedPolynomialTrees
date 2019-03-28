#ifndef DATASETHANDLER_H
#define	DATASETHANDLER_H

#include "model/Types.h"
#include "model/detectors/Detector.h"

#include <log4cxx/logger.h>

namespace faceAnalysis {
  class DatasetHandler {
    static log4cxx::LoggerPtr logger;

  public:
    explicit DatasetHandler(std::string cascade);
    DatasetHandler();

    ~DatasetHandler() {}


    cv::Rect ExtractFaceBox(const DatasetExample training_example);
    std::vector<cv::Rect> ExtractFaceBoxes(Dataset evaluation_data);

    std::vector<std::string> GenerateSamples(
        const int number,const std::vector<std::string> list_of_all_files);

    std::vector<cv::Rect> ReadBoundingBoxes(
        const std::vector<std::string> readed_data_names,
        const std::string path_to_bounding_boxes, const bool using_ground_truth);

    std::vector<std::string> ReadFileNames(
        const std::string path_to_file_containing_names);

    Dataset ReadDataset(const std::vector<std::string> readed_data_names,
                        const std::string path_to_point_folder,
                        const std::string path_to_image_folder,
                        const DataBaseType database_type);

    Dataset TransformTraining(const Dataset training_data,
                              const std::string path_to_bounding_boxes,
                              const std::vector<std::string> list_of_all_files);

    void WriteUsedFileNames(const std::string name,
                            const std::vector<std::string> list_of_used_files);

    cv::Mat GrayScaleEnergyNormalization(const cv::Mat image_part);

  private:
    Dataset ReadMultipieDataset(const std::vector<std::string> readed_data_names,
                                const std::string path_to_point_folder,
                                const std::string path_to_image_folder);

    Dataset ReadOtherDataset(const std::vector<std::string> readed_data_names,
                             const std::string path_to_point_folder,
                             const std::string path_to_image_folder,
                             const std::string suffix);


    DetectorPtr detector_;
    std::string cascade_;
  };
}

#endif	/* DATASETHANDLER_H */

