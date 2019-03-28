#include <boost/filesystem.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>

#include <tclap/CmdLine.h>

#include "model/evaluation/TotalEvaluator.h"

int main(int argc, char **argv) {
  using std::map;
  using std::string;
  using std::vector;

  //  QApplication a(argc, argv);
  //  MainWindow window;
  //  window.show();
  //  return a.exec();

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
    cmd.add(configFileArg);
    cmd.parse(argc, argv);

    // parse config file
    boost::property_tree::ptree configuration;
    boost::property_tree::ini_parser::read_ini(configFileArg.getValue(),
                                               configuration);

    // creating logger
    string loggerPath = configuration.get<std::string>("logger.path");
    log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("CPT.mainEvaluation"));
    if (boost::filesystem::exists(loggerPath))
      {
        log4cxx::PropertyConfigurator::configure(loggerPath);
        LOG4CXX_INFO(logger, "Using logger config file " << loggerPath);
      } else {
        log4cxx::BasicConfigurator::configure();
        LOG4CXX_INFO(logger, "Using basic logger");
      }

    // creating evaluator
    faceAnalysis::TotalEvaluator evaluator;
    evaluator.Init(configuration);
    evaluator.Evaluate();

  } catch (TCLAP::ArgException &e)  {
    std::cerr <<  e.error() << " for arg " << e.argId() << std::endl;
    return -1;
  }
}
