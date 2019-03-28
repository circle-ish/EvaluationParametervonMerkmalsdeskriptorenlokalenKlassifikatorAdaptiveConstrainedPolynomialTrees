#include "model/trainers/Trainer.h"
#include "model/local/GlobalDefinitions.h"

#include "model/CmdInputMapping.h"
#include "model/dataHandlers/DatasetHandler.h"
#include "model/global/GlobalTree.h"
#include "model/local/LocalPolynoms.h"

#include <boost/filesystem.hpp>

log4cxx::LoggerPtr faceAnalysis::Trainer::logger(
    log4cxx::Logger::getLogger("CPT.model.Trainer"));

faceAnalysis::Trainer::Trainer(
        boost::property_tree::ptree configuration,
        std::vector<int> landmarksToModel
        )

        : landmarks_to_model_(landmarksToModel),
          configuration_(configuration)
        {

    output_folder_path_ = configuration.get<std::string>("training.outputFolder");

    if (!boost::filesystem::exists(output_folder_path_)) {
        LOG4CXX_ERROR(logger, "Output folder does not exist!");
        exit(-1);
    }

    InitTraining(configuration);
}

void faceAnalysis::Trainer::call() {
    std::stringstream s, in;
    s << "/" << "pr_" << PATCH_RADIUS;
    if ((useFeature & (USE_HOG | USE_COHOG)) != 0) {
        s << "_cs" << "_" << cellSize.height << "x" << cellSize.width << "_"
          << "bs" << "_" << blockSize.height << "x" << blockSize.width << "_"
          << "bo" << "_" << blockOverlap.height << "x" << blockOverlap.width << "_"
          << "bn" << "_" << binNo << "_" << "us" << "_" << useSign;
    }
    if (APPLY_PCA) {
        s << "_pca";
    }
    if ((useFeature & USE_HOG) != 0) {
        s << "_" << "HOG";
    }
    if ((useFeature & USE_COHOG) != 0) {
        s << "_" << "on" << "_" << offsetNo
          << "_" << "COHOG";
    }
    if ((useFeature & USE_MRLBP) != 0) {
        s << "_rn_" << ringNo
          << "_" << "MRLBP";
    }
    if ((useFeature & USE_LBP) != 0) {
        s << "_" << "LBP";
    }
    if ((useFeature & USE_GRAD) != 0) {
        s << "_" << "GRAD";
    }
    configuration_.put("training.subFolder", s.str().c_str());

    try {
        Trainer::TrainLocal();
    } catch (std::string message) {
        return;
    }
}

void faceAnalysis::Trainer::TrainLocalWrapper() {
    /** only one param is changed at a time;
     * default values; see GlobalDefinitions.h */
    cellSize = cv::Size(3,3);
    blockSize = cv::Size(4, 4);
    blockOverlap = cv::Size(0, 0);
    binNo = 6;
    useSign = false;
    offsetNo = 4;

    ringNo = 2;
    half = true;

    PATCH_RADIUS = 6;
    PATCH = PATCH_RADIUS * 2;
    PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);

    retainVal = 0.85;

    verbosity = 0;

    const int maxCellSize = 16;
    const int stepCellSize = 4;
    const int maxBlockSize = 6;
    const int stepBlockSize = 2;
    const int maxBlockOverlap = 3;
    const int stepBlockOverlap = 1;
    const int maxBinNo = 16;
    const int stepBinNo = 2;
    const int maxOffsetNo = 24;
    const int stepOffsetNo = 3;

    if (faceAnalysis::call) {
        Trainer::call();
        return;
    }

    int methodNo = 0;
    if ((useFeature & USE_HOG) != 0) {
        ++methodNo;
    }
    if ((useFeature & USE_COHOG) != 0) {
        ++methodNo;
    }
    if ((useFeature & USE_MRLBP) != 0) {
        ++methodNo;
    }
    if ((useFeature & USE_LBP) != 0) {
        ++methodNo;
    }
    if ((useFeature & USE_GRAD) != 0) {
        ++methodNo;
    }
    //if (methodNo > 1) {
        call();
        return;
    //}

    int i;
    cv::Size tmp;

    i = PATCH_RADIUS;
    for (; p < 20; p += 2) {
        PATCH_RADIUS = p;
        PATCH = PATCH_RADIUS * 2;
        PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);

        call();
    }
    PATCH_RADIUS = i;
    PATCH = PATCH_RADIUS * 2;
    PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);

    if ((useFeature & USE_HOG) != 0) {
        cohog = false;
    }
    if ((useFeature & USE_COHOG) != 0) {
        cohog = true;
        tmp = blockOverlap;
        blockOverlap = cv::Size(0, 0);
    }
    if ((useFeature & (USE_HOG | USE_COHOG)) != 0) {
        tmp = cellSize;
        for (; a < maxCellSize + 1; a += stepCellSize) {
            cellSize = cv::Size(a, a);
            call();
        }
        cellSize = tmp;

        tmp = cellSize;
        for (; a1 < maxCellSize + 1; a1 += stepCellSize) {
            for (; b1 < maxCellSize + 1; b1 += stepCellSize) {
                PATCH_RADIUS = 2 * std::max(a1, b1);
                PATCH = PATCH_RADIUS * 2;
                PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);
                cellSize = cv::Size(a1, b1);
                call();
            }
        }
        cellSize = tmp;
        PATCH_RADIUS = 8;
        PATCH = PATCH_RADIUS * 2;
        PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);

        cv::Size tmp2 = blockOverlap;
        tmp = blockSize;
        for (; b < maxBlockSize + 1; b += stepBlockSize) {
            blockSize = cv::Size(b, b);
            if ((useFeature & USE_HOG) != 0) {
                for (; c < std::min(blockSize.height, maxBlockOverlap + 1); c += stepBlockOverlap) {
                    blockOverlap = cv::Size(c, c);
                    call();
                }
            } else {
                call();
            }
        }
        blockSize = tmp;
        blockOverlap = tmp2;

        i = binNo;
        for (; d < maxBinNo + 1; d += stepBinNo) {
            binNo = d;
            call();
        }
        binNo = i;
    }

    if ((useFeature & USE_COHOG) != 0) {
        i = offsetNo;
        cellSize = cv::Size(8, 8);
        blockSize = cv::Size(2, 2);
        PATCH_RADIUS = 12;
        PATCH = PATCH_RADIUS * 2;
        PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);
        for (; e < maxOffsetNo + 1; e += stepOffsetNo) {
            offsetNo = e;
            call();
        }
    }

    if ((useFeature & USE_MRLBP) != 0) {
        call();
    }
}

void faceAnalysis::Trainer::TrainLocal() {
  LocalPolynomsPtr local_trainer = LocalPolynomsPtr(
        new LocalPolynoms(configuration_, landmarks_to_model_));
  try {
      local_trainer->Train(data_);
  } catch (std::string message) {
      LOG4CXX_INFO(logger, message);
      throw message;
  }
}

void faceAnalysis::Trainer::TrainGlobal() {
  GlobalTreePtr global_model = GlobalTreePtr(new GlobalTree());
  global_model->set_landmarks_to_model(landmarks_to_model_);
  global_model->Train(data_);

  global_model_file_name_ = configuration_.get<std::string>("training.globalModelFileName");
  std::stringstream tree_path_stream;
  tree_path_stream << output_folder_path_ << "/" << global_model_file_name_;
  global_model->Save(tree_path_stream.str());

  global_model->PrintOutput("tree.dot");
}


void faceAnalysis::Trainer::InitTraining(
    const boost::property_tree::ptree configuration) {
  DatasetHandler data_handler(
        configuration.get<std::string>("detector.cascade"));

  std::string point_folder_path =
      configuration.get<std::string>("training.ptrFolder");
  if (!boost::filesystem::exists(point_folder_path))
    {
      LOG4CXX_ERROR(logger, "Point folder does not exist!");
      exit(-1);
    }


  if (!configuration.get<bool>("training.readImages"))
    {
      // TODO das Laden/Speichern hier ist auch nicht ganz optimal....
      std::vector<std::string> all_files_list =
          data_handler.ReadFileNames(point_folder_path + "/files.txt");

      std::string images_to_use_string =
          configuration.get<std::string>("training.imagesToUse");
      int random_example_number =
          configuration.get<int>("training.randExampleNumber");

      std::vector<std::string> used_file_list;
      switch (faceAnalysis::CmdInputMapping::GetImagesToUseType(
                images_to_use_string))
        {
        case FIRST:
          used_file_list.push_back(all_files_list.at(0));
          break;
        case ALL:
          used_file_list = all_files_list;
          break;
        case RANDOM:
          used_file_list =
              data_handler.GenerateSamples(
                random_example_number, all_files_list);
          break;
        default:
          LOG4CXX_WARN(
                logger, "Unknown type " << images_to_use_string << " -- EXIT");
          exit(-1);

        }

      data_handler.WriteUsedFileNames(output_folder_path_ + "/data.txt",
                                      used_file_list);
    }

  std::vector<std::string> readed_data =
      data_handler.ReadFileNames(output_folder_path_ + "/data.txt");

  std::string img_folder_path =
      configuration.get<std::string>("training.imgFolder");
  if (!boost::filesystem::exists(img_folder_path))
    {
      LOG4CXX_ERROR(logger, "Image folder does not exist!");
      exit(-1);
    }
  if (!boost::filesystem::is_directory(img_folder_path))
    {
      LOG4CXX_ERROR(logger, "Image folder is not a (valid) folder!");
      exit(-1);
    }

  std::string database = configuration.get<std::string>("training.databaseName");
  DataBaseType database_type = CmdInputMapping::GetDataBaseType(database);
  if (database_type == faceAnalysis::DATABASETYPE_INIT)
    {
      LOG4CXX_ERROR(logger, "unknown database type " << database << " -- EXIT");
      exit(-1);
    }


  data_ = data_handler.ReadDataset(readed_data, point_folder_path,
                                   img_folder_path, database_type);

  std::string bounding_boxes_folder_path =
      configuration.get<std::string>("training.bbFolder");
  data_ = data_handler.TransformTraining(data_,
                                         bounding_boxes_folder_path,
                                         readed_data);

  if (landmarks_to_model_.empty())
    {
      LOG4CXX_INFO(logger, "using all landmarks");
      for (int i = 0; i < data_[0].second.rows; i++)
        {
          landmarks_to_model_.push_back(i);
        }
    }
}
