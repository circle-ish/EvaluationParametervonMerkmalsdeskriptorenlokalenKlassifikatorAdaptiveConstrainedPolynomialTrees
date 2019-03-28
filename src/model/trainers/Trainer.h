#ifndef TRAINER_H
#define	TRAINER_H

#include <boost/shared_ptr.hpp>
#include <boost/property_tree/ptree.hpp>

#include <log4cxx/logger.h>

#include "model/Types.h"

namespace faceAnalysis {
  class Trainer {
    static log4cxx::LoggerPtr logger;
  public:
    explicit Trainer(boost::property_tree::ptree configuration,
                     std::vector<int> landmarksToModel);
    ~Trainer() {}

    void TrainLocalWrapper();
    void call();
    void TrainLocal();
    void TrainGlobal();

  private:
    void InitTraining(const boost::property_tree::ptree configuration);

    std::vector<int> landmarks_to_model_;
    std::string output_folder_path_;
    std::string global_model_file_name_;
    Dataset data_;
    boost::property_tree::ptree configuration_;

  };
  typedef boost::shared_ptr<Trainer> TrainerPtr;
}


#endif	/* TRAINER_H */

