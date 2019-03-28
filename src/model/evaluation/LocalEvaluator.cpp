#include "LocalEvaluator.h"
#include "model/local/GlobalDefinitions.h"
#include "model/GlobalConstants.h"

log4cxx::LoggerPtr faceAnalysis::LocalEvaluator::logger(
    log4cxx::Logger::getLogger("CPT.models.evaluation.localEvaluator"));

faceAnalysis::LocalEvaluator::LocalEvaluator(){}
faceAnalysis::LocalEvaluator::~LocalEvaluator(){}

void faceAnalysis::LocalEvaluator::call(
        const std::vector<int> &landmarks_to_evaluate
        ) {

    /** create identifier and paths for current configuration */
    std::stringstream s, out;
    s << "/" << "pr_" << PATCH_RADIUS;
    if ((useFeature & (USE_HOG | USE_COHOG)) != 0) {
        s << "_cs_" << cellSize.height << "x" << cellSize.width << "_"
          << "bs_" << blockSize.height << "x" << blockSize.width << "_"
          << "bo_" << blockOverlap.height << "x" << blockOverlap.width << "_"
          << "bn_" << binNo << "_" << "us_" << useSign;
    }
    if (APPLY_PCA) {
        s << "_pca";
    }
    if ((useFeature & USE_HOG) != 0) {
        s << "_" << "HOG";
    }
    if ((useFeature & USE_COHOG) != 0) {
        s << "_" << "on_" << offsetNo
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

    out << configuration_.get<std::string>("evaluation.outputFolder") << s.str();
    outputFile = out.str();

    LOG4CXX_INFO(logger, "_______________________________evaluation"
                 << "_________________________________\nParams: " << s.str());

    /** evaluate and measure distance between trained and actual landmarks */
    std::vector<double> hitMissRate;
    hitMissRate.resize(68);
    std::vector<double> hitMissRateSum;
    hitMissRateSum.resize(68);
    try {
        LocalEvaluator::Evaluate(landmarks_to_evaluate, hitMissRate, hitMissRateSum);
    } catch (std::string message) {
        return;
    }

    out << "/hitmiss";
    std::ofstream output;
    output.open(out.str().c_str(), std::fstream::out | std::fstream::app);

    for (
         std::vector<int>::const_iterator it = landmarks_to_evaluate.begin();
         it != landmarks_to_evaluate.end();
         ++it
         ) {

        output << (*it) << "=" << hitMissRate.at(*it) << std::endl;
        output << (*it) << "=" << hitMissRateSum.at(*it) << std::endl;
    }

    output.close();
}

void faceAnalysis::LocalEvaluator::EvaluateWrapper(
        const std::vector<int> &landmarks_to_evaluate
        ) {

    /** only one param is changed at a time;
     * default values; see GlobalDefinitions.h */
    cellSize = cv::Size(2, 3);
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
        LocalEvaluator::call(landmarks_to_evaluate);
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
//    if (methodNo > 1) {
        call(landmarks_to_evaluate);
        return;
    //}

    int i;
    cv::Size tmp;

    i = PATCH_RADIUS;
    for (; p < 20; p += 2) {
        PATCH_RADIUS = p;
        PATCH = PATCH_RADIUS * 2;
        PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);

        call(landmarks_to_evaluate);
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
            call(landmarks_to_evaluate);
        }
        cellSize = tmp;

        tmp = cellSize;
        for (; a1 < maxCellSize + 1; a1 += stepCellSize) {
            for (; b1 < maxCellSize + 1; b1 += stepCellSize) {
                PATCH_RADIUS = 2 * std::max(a1, b1);
                PATCH = PATCH_RADIUS * 2;
                PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);
                cellSize = cv::Size(a1, b1);
                call(landmarks_to_evaluate);
            }
        }
        cellSize = tmp;
        PATCH_RADIUS = 8;
        PATCH = PATCH_RADIUS * 2;
        PATCH_LEFT_HALF = (PATCH_RADIUS == std::ceil(PATCH / (double)2) ? PATCH_RADIUS - 1 : PATCH_RADIUS);

        tmp = blockSize;
        cv::Size tmp2 = blockOverlap;
        for (; b < maxBlockSize + 1; b += stepBlockSize) {
            blockSize = cv::Size(b, b);
            if ((useFeature & USE_HOG) != 0) {
                for (; c < std::min(blockSize.height, maxBlockOverlap + 1); c += stepBlockOverlap) {
                    blockOverlap = cv::Size(c, c);
                    call(landmarks_to_evaluate);
                }
            } else {
                call(landmarks_to_evaluate);
            }
        }
        blockSize = tmp;
        blockOverlap = tmp2;

        i = binNo;
        for (; d < maxBinNo + 1; d += stepBinNo) {
            binNo = d;
            call(landmarks_to_evaluate);
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
            call(landmarks_to_evaluate);
        }
    }

    if ((useFeature & USE_MRLBP) != 0) {
        call(landmarks_to_evaluate);
    }
}

double faceAnalysis::LocalEvaluator::Evaluate(
        const std::vector<int> &landmarks_to_evaluate,
        std::vector<double> &hitMissRate,
        std::vector<double> &hitMissRateSum
        ) {

    /** load Polynoms */
  std::stringstream loadPath;
  loadPath << path_to_trained_model_ << configuration_.get<std::string>("training.subFolder") << "/";
  boost::filesystem::path dir(loadPath.str());
  if (!(boost::filesystem::exists(dir))) {
      std::stringstream error;
      error << "requested polynom directory " << loadPath.str() << " does not exist";
      LOG4CXX_ERROR(logger, error.str());
      throw error.str();
  }

  LocalPolynomsPtr localPolynoms(faceAnalysis::LocalPolynomsPtr(new faceAnalysis::LocalPolynoms(configuration_)));
  localPolynoms->Load(loadPath.str().c_str());

  std::vector<int> landmarks_to_use;
  if (landmarks_to_evaluate.empty()) {
      for (int i = 0; i < localPolynoms->Size(); i++ ) {
          landmarks_to_use.push_back(i);
      }
  } else {
      landmarks_to_use = landmarks_to_evaluate;
  }

  boost::filesystem::path outDir(outputFile);
  if (!(boost::filesystem::exists(outDir) || boost::filesystem::create_directory(outDir))) {
      LOG4CXX_ERROR(logger, "Could not create directory " << outputFile);
  }

  int counter = 0;
  for (
       Dataset::const_iterator itr = dataset_.begin() + counter;
       itr != dataset_.end();
       ++itr
       ) {

//      if (counter == 5) break;
      no = counter;

      LOG4CXX_INFO(logger, "Image " << counter << "/" << dataset_.size());
      DatasetExample ground_truth = *itr;
      cvtColor(ground_truth.first, ground_truth.first, CV_RGB2GRAY);

      /** scale face to determined width */
      cv::Rect detector_output = bounding_boxes_.at(counter);

      double scale_factor = NORM_FACE_WIDTH / (double)(PATCH + detector_output.width);
      cv::Mat scaled_image;
      resize(ground_truth.first, scaled_image,
             cv::Size(scale_factor * ground_truth.first.cols,
                      scale_factor * ground_truth.first.rows));
      cv::Rect scaled_rect;
      scaled_rect.x = std::max(0, (int)round(scale_factor * detector_output.x) - PATCH_LEFT_HALF);
      scaled_rect.y = std::max(0, (int)round(scale_factor * detector_output.y) - PATCH_LEFT_HALF);
      scaled_rect.width = std::min((int)NORM_FACE_WIDTH + scaled_rect.x, scaled_image.cols) - scaled_rect.x;
      scaled_rect.height = std::min((int)round(scale_factor * detector_output.height) + scaled_rect.y, scaled_image.rows) - scaled_rect.y;
      cv::Mat detection_image = scaled_image(scaled_rect);
//      cv::Mat detection_image = ground_truth.first(detector_output);

      if (detection_image.cols != (int)NORM_FACE_WIDTH) {
          LOG4CXX_INFO(logger, "Smaller face rect than usual" << detection_image.size());
      }

      /** generate features and retrieve answer values */
      if (verbosity > 3) LOG4CXX_INFO(logger, "Generate local features");
      localPolynoms->InitFeatureMap(detection_image.rows, detection_image.cols);
      localPolynoms->SetCurrentImage(detection_image);

      std::vector<std::vector<double> > features;
      try {
          localPolynoms->ExtractFeatures(features);
      } catch (std::string message) {
          std::cout << message << std::endl;
          continue;
      }
      if (verbosity > 3) LOG4CXX_INFO(logger, "Finished to generate local features");

      // Create a vector containing the channels of the new colored image
      std::vector<cv::Mat> channels;
      channels.push_back(ground_truth.first); // 1st channel
      channels.push_back(ground_truth.first); // 2nd channel
      channels.push_back(ground_truth.first); // 3rd channel
      // Construct a new 3-channel image of the same size and depth
      cv::Mat gui;
      cv::merge(channels, gui);

      /** for coloring landmarks in different colors */
      int rbg[3] = {0, 0, 0};
      const int colorStep = 750 / landmarks_to_use.size();
      int lmCounter = 0;

      for(
          std::vector<int>::const_iterator it = landmarks_to_use.begin();
          it != landmarks_to_use.end();
          ++it
          ) {

          int polyDegree = 2;
          localPolynoms->GenerateFeatureMap(features, polyDegree, *it, loadPath.str());
          if (verbosity > 3) LOG4CXX_INFO(logger, "Evaluating polynom " << *it);
          if (!localPolynoms->IsAnswerMapInitialized(*it)) {
              localPolynoms->initAnswer(*it);
          }

          for(int col_itr=0; col_itr < detection_image.cols; col_itr++) {
              for(int row_itr = 0; row_itr < detection_image.rows; row_itr++) {
                  if (localPolynoms->getAnswer(*it,row_itr, col_itr) == 0) {
                      localPolynoms->GenerateAnswerForPolynom(*it, row_itr, col_itr);
                  }
              }
          }
          localPolynoms->GenerateSumAnswerForPolynom(*it);

          if (verbosity > 3) LOG4CXX_INFO(logger, "Draw heatmap");
//#if SHOW_IMAGES == 1
          /*if (verbosity > 3) */localPolynoms->DrawAnswerMap(*it);
//#endif

          // Ground truth
          cv::Point2i ground_truth_point(ground_truth.second.at<float>(*it, 0),
                                         ground_truth.second.at<float>(*it, 1));
          cv::Point2i max_local_answer = localPolynoms->GetMaxValuePosition(*it);
          cv::Point2i max_sum_answer = localPolynoms->GetMaxValuePositionBySum(*it);

          /** calculate distance between found and actual landmark;
            * sum over all images */
          hitMissRate.at(*it) += (
                      std::sqrt(
                          std::pow(ground_truth_point.x - max_local_answer.x, 2)
                          + std::pow(ground_truth_point.y - max_local_answer.y, 2)
                          ) / dataset_.size()
                      );
          hitMissRateSum.at(*it) += (
                      std::sqrt(
                          std::pow(ground_truth_point.x - max_sum_answer.x, 2)
                          + std::pow(ground_truth_point.y - max_sum_answer.y, 2)
                          ) / dataset_.size()
                      );
          //rescale to init image
          max_local_answer.x = max_local_answer.x / scale_factor
              + detector_output.x;
          max_local_answer.y = max_local_answer.y / scale_factor
              + detector_output.y;
          max_sum_answer.x = max_sum_answer.x / scale_factor
              + detector_output.x;
          max_sum_answer.y = max_sum_answer.y / scale_factor
              + detector_output.y;


          /** different colors for every found point; annotated points are white*/
//#if SHOW_IMAGES == 1
          int i = lmCounter * colorStep / 250;
          ++lmCounter;
          rbg[i] += colorStep;
          rbg[(i + 1) % 3] = 0;
          rbg[(i + 2) % 3] = 0;
          cv::rectangle(gui, detector_output, cv::Scalar(0,0,0),2);
          cv::circle(gui, max_local_answer,1,cv::Scalar(255, 0, 0),2);
          cv::circle(gui, max_sum_answer,1,cv::Scalar(0, 0, 255),2);
          cv::circle(gui, ground_truth_point,1,cv::Scalar(255, 255, 255),1);
          cv::line(gui, ground_truth_point, max_local_answer, cv::Scalar(rbg[0], rbg[1], rbg[2]));
          cv::line(gui, ground_truth_point, max_sum_answer, cv::Scalar(rbg[0], rbg[1], rbg[2]));
//#endif
        }

      cv::Mat tmp;
      faceAnalysis::PatchFeatureUtils::zoomMatrix<cv::Vec3b>(gui(detector_output), tmp, 2);

//      std::stringstream s;
//      s << outputFile << "/face" << no << ".jpg";
//      cv::imwrite(s.str(), tmp);
//#if SHOW_IMAGES == 1
      cv::imshow("Unscaled Face", tmp);
      cv::waitKey(0);
//#endif
      counter++;

      localPolynoms->FreeAllPolynoms();
      localPolynoms->FreeFeatureMap();
    }

  //TODO return something
  return -1;
}

double faceAnalysis::LocalEvaluator::Evaluate(
        const std::vector<int> &landmarks_to_evaluate
        ) { return -1; }

