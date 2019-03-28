#ifndef DAIMLERDATAHANDLER_H
#define	DAIMLERDATAHANDLER_H

#include "model/Types.h"

#include <vector>
#include <string>

#include <log4cxx/logger.h>

namespace faceAnalysis {

  class DaimlerDataHandler  {
    static log4cxx::LoggerPtr logger;
  public:
    explicit DaimlerDataHandler(const std::string folder_path,
                                const std::string camera,
                                const ClassifierType classifier,
                                const std::string clm_data_path);
    ~DaimlerDataHandler() {}

    std::vector<std::string> GenerateCsv(const std::string cascade,
                                         const bool wait);
    void SaveFile(const std::vector<std::string> points);

  private:
    cv::Mat ConvertImage(const cv::Mat raw_image);
    std::vector<bool> GenerateFrameDropList();

    std::string folder_path_;
    std::string camera_;
    ClassifierType classifier_;
    std::string clm_data_path_;



  };
}
#endif	/* DAIMLERDATAHANDLER_H */

