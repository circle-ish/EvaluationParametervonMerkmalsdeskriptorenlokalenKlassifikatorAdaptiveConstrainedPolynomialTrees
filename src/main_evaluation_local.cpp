#include <boost/filesystem.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>

#include <tclap/CmdLine.h>
#include "model/Types.h"
#include "model/local/GlobalDefinitions.h"

#include "model/evaluation/LocalEvaluator.h"

/** see GlobalDefinitions.h */
int faceAnalysis::p, faceAnalysis::a, faceAnalysis::b, faceAnalysis::a1;
int faceAnalysis::b1, faceAnalysis::c, faceAnalysis::d, faceAnalysis::e;
int faceAnalysis::useFeature;
bool faceAnalysis::APPLY_PCA;
bool faceAnalysis::call;

int main(int argc, char **argv) {
  using std::map;
  using std::string;
  using std::vector;

  // Create and parse command line
  try {
    static const string commandLineHelpOutput =
        "Evaluate polynomial CLM.\n "
        "Example: "
        "./CPT_Evaluation"
        "-c /homes/ttoenige/NetBeansProjects/agaifaceanalysis.git.cpt/configEvaluation.ini";

    TCLAP::CmdLine cmd(commandLineHelpOutput, ' ', "0.1", true);
    TCLAP::ValueArg<string> configFileArg("c", "config", "path to config file",
                                          true, "", "path-to-config");
    TCLAP::ValueArg<int> radArg("p", "patch_radius", "", false, 2, "", cmd);
    TCLAP::ValueArg<int> csArg("a", "cellSize", "", false, 4, "", cmd);
    TCLAP::ValueArg<int> bsArg("b", "blockSize", "", false, 2, "", cmd);
    TCLAP::ValueArg<int> csVArg("v", "cell_v", "", false, 2, "", cmd);
    TCLAP::ValueArg<int> csHArg("z", "cell_h", "", false, 2 + 1, "", cmd);
    TCLAP::ValueArg<int> boArg("o", "blockOverlap", "", false, 0, "", cmd);
    TCLAP::ValueArg<int> bnArg("d", "binNo", "", false, 4, "", cmd);
    TCLAP::ValueArg<int> ofArg("e", "offsetNo", "", false, 6, "", cmd);
    TCLAP::ValueArg<std::string> methodArg("m", "method", "", true, "", "", cmd);
    TCLAP::SwitchArg pcaArg("r", "pca", "", cmd, false);
    TCLAP::SwitchArg callArg("s", "call_once", "", cmd, false);
    cmd.add(configFileArg);
    TCLAP::UnlabeledMultiArg<int> landmarksToModelArg(
          "landmarksToEvaluate",
          "landmarks to evaluate (if flag not present --> use all landmarks)",
          false, "landmarks-to-model");
    cmd.add(landmarksToModelArg);
    cmd.parse(argc, argv);

    vector<int> landmarksToEvaluate = landmarksToModelArg.getValue();


    // parse config file
    boost::property_tree::ptree configuration;
    boost::property_tree::ini_parser::read_ini(configFileArg.getValue(),
                                               configuration);

    // creating logger
    log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("CPT.mainEvaluation"));
    string loggerPath = configuration.get<std::string>("logger.path");
    if (boost::filesystem::exists(loggerPath))
      {
        log4cxx::PropertyConfigurator::configure(loggerPath);
        LOG4CXX_INFO(logger, "Using logger config file " << loggerPath);
      } else {
        log4cxx::BasicConfigurator::configure();
        LOG4CXX_INFO(logger, "Using basic logger");
      }

    // creating evaluator
    faceAnalysis::LocalEvaluator evaluator;
    evaluator.Init(configuration);
    faceAnalysis::useFeature = 0;
    faceAnalysis::p = radArg.getValue();
    faceAnalysis::a = csArg.getValue();
    faceAnalysis::b = bsArg.getValue();
    faceAnalysis::a1 = csVArg.getValue();
    faceAnalysis::b1 = csHArg.getValue();
    faceAnalysis::c = boArg.getValue();
    faceAnalysis::d = bnArg.getValue();
    faceAnalysis::e = ofArg.getValue();
    faceAnalysis::APPLY_PCA = pcaArg.getValue();
    faceAnalysis::call = callArg.getValue();
    std::string method = methodArg.getValue();
    if (method.find("h") != std::string::npos) {
        faceAnalysis::useFeature += faceAnalysis::USE_HOG;
    }
    if (method.find("c") != std::string::npos) {
        faceAnalysis::useFeature += faceAnalysis::USE_COHOG;
    }
    if (method.find("m") != std::string::npos) {
        faceAnalysis::useFeature += faceAnalysis::USE_MRLBP;
    }
    if (method.find("g") != std::string::npos) {
        faceAnalysis::useFeature += faceAnalysis::USE_GRAD;
    }
    if (method.find("l") != std::string::npos) {
        faceAnalysis::useFeature += faceAnalysis::USE_LBP;
    }
    if (faceAnalysis::useFeature == 0) {
        return -1;
    }

    evaluator.EvaluateWrapper(landmarksToEvaluate);

  } catch (TCLAP::ArgException &e)  {
    std::cerr <<  e.error() << " for arg " << e.argId() << std::endl;
    return -1;
  }
}
