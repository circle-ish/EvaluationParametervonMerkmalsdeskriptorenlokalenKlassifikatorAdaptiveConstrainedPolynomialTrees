#include <boost/filesystem.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tclap/CmdLine.h>

#include "model/CmdInputMapping.h"
#include "model/Types.h"
#include "model/trackers/CPT.h"
#include "model/detectors/ViolaLienhartDetector.h"


int main(int argc, char **argv) {
  using std::map;
  using std::string;

  // Create and parse command line
  try {
    static const string commandLineHelpOutput =
        "Tracking face from video/image folder/camera\n "
        "Example: "
        "./CPT_Track"
        "-c /homes/ttoenige/NetBeansProjects/agaifaceanalysis.git.cpt/configTraining.ini";

    TCLAP::CmdLine cmd(commandLineHelpOutput, ' ', "0.1", true);
    TCLAP::ValueArg<string> configFileArg("c", "config", "path to config file",
                                          true, "", "path-to-config");
    cmd.add(configFileArg);
    cmd.parse(argc, argv);

    // parse config file
    boost::property_tree::ptree configuration;
    boost::property_tree::ini_parser::read_ini(configFileArg.getValue(),
                                               configuration);


    // creating logger
    string loggerPath = configuration.get<std::string>("logger.path");
    log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("CPT.main"));
    if (boost::filesystem::exists(loggerPath)) {
        log4cxx::PropertyConfigurator::configure(loggerPath);
        LOG4CXX_INFO(logger, "Using logger config file " << loggerPath);
      } else {
        log4cxx::BasicConfigurator::configure();
        LOG4CXX_INFO(logger, "Using basic logger");
      }


    string outputFolderPath =
        configuration.get<std::string>("training.outputFolder");

    faceAnalysis::CPTPtr cpt = faceAnalysis::CPTPtr(
          new faceAnalysis::CPT(configuration, std::vector<int>()));
    cpt->Load(outputFolderPath);

    string cascade = configuration.get<std::string>("detector.cascade");
    faceAnalysis::DetectorPtr detector =
        faceAnalysis::DetectorPtr(
          new faceAnalysis::ViolaLienhartDetector(cascade));

    string trackingInput = configuration.get<std::string>("tracking.input");
    cv::VideoCapture cap(trackingInput);
    if (!cap.isOpened()) {
        LOG4CXX_ERROR(logger, "Cannot open as string, trying as int");
        cap.open(atoi(trackingInput.c_str()));

        if (!cap.isOpened()) {
            LOG4CXX_ERROR(logger, "Still not working -- exit");
            return -1;
          }
      }

    cv::Mat frame;
    cv::Rect detectorOutput;
    for (;;) {
        cap >> frame;

        cv::cvtColor(frame, frame, CV_RGB2GRAY);

        // get biggest face
        std::map<int, cv::Rect> detection = detector->Detect(frame);
        double maxArea = std::numeric_limits<double>::min();
        int maxID = -1;
        for (uint i = 0; i < detection.size(); i++) {
            if (detection[i].area() > maxArea) {
                maxArea = detection[i].area();
                maxID = i;
              }
          }

        if (detection.size() != 0) {
            detectorOutput = detection[maxID];

            cpt->FreeBc();
            cpt->FreeBcPosition();
            cpt->ClearPoints();
            cpt->FreePolynoms();

            cpt->DetectFeatures(true);
          }

        cv::waitKey(1);

        switch (cv::waitKey(10)) {
          case 'r':
            break;
          case 27:
            LOG4CXX_INFO(logger, "ESC");
            exit(0);
            break;
          }

      }


  } catch (TCLAP::ArgException &e) {
    std::cerr <<  e.error() << " for arg " << std::endl;
    return 1;
  }
}
