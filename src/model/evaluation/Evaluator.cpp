#include "model/evaluation/Evaluator.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

#include "model/dataHandlers/DatasetHandler.h"
#include "model/CmdInputMapping.h"
#include "model/GlobalConstants.h"

log4cxx::LoggerPtr faceAnalysis::Evaluator::logger(
    log4cxx::Logger::getLogger("CPT.models.evaluation.evaluator"));

void faceAnalysis::Evaluator::GenerateEvaluationData() {
    database_to_use_ = faceAnalysis::CmdInputMapping::GetDataBaseType(database_to_use_string_);
    if (database_to_use_ == faceAnalysis::DATABASETYPE_INIT) {
        LOG4CXX_WARN(logger, "unknown database type " << database_to_use_string_
                   << " -- EXIT");
        exit(-1);
    }

    // read evaluation data
    faceAnalysis::DatasetHandler datahandler;
    std::vector<std::string> list_of_all_data = datahandler.ReadFileNames(path_to_point_folder_ + "/files.txt");
    dataset_ = datahandler.ReadDataset(
                list_of_all_data,
                path_to_point_folder_,
                path_to_images_,
                database_to_use_
                );

    // read bounding boxes
    if (path_to_bounding_boxes_.empty()) {
        LOG4CXX_DEBUG(logger, "Bounding boxes not provided in the database -- generate them");
        bounding_boxes_ = datahandler.ExtractFaceBoxes(dataset_);
    } else {
        LOG4CXX_DEBUG(logger, "Bounding boxes provided by database -- read them");
        bounding_boxes_ = datahandler.ReadBoundingBoxes(
                    list_of_all_data,
                    path_to_bounding_boxes_,
                    true
                    );
    }
}

void faceAnalysis::Evaluator::Init(const boost::property_tree::ptree configuration) {
    configuration_ = configuration;
    path_to_bounding_boxes_ = configuration.get<std::string>("evaluation.bbFolder");
    path_to_point_folder_   = configuration.get<std::string>("evaluation.ptrFolder");
    path_to_images_         = configuration.get<std::string>("evaluation.imgFolder");
    path_to_trained_model_  = configuration.get<std::string>("training.outputFolder");
    database_to_use_string_ = configuration.get<std::string>("evaluation.databaseName");

    GenerateEvaluationData();
}
