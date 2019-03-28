#ifndef INPUTMAPPING_H
#define INPUTMAPPING_H

#include "Types.h"

namespace faceAnalysis
{
  class CmdInputMapping
  {
  public:
    static DataBaseType GetDataBaseType(std::string database_string)
    {
      // convert from string to enum
      DataBaseType database_type = faceAnalysis::DATABASETYPE_INIT;

      if (database_string == "multipie")
        database_type = MULTIPIE;
      else if (database_string == "franck")
        database_type = FRANCK;
      else if (database_string == "afw")
        database_type = AFW_300W;
      else if (database_string == "helen_test")
        database_type = HELEN_TEST_300W;
      else if (database_string == "helen_train")
        database_type = HELEN_TRAIN_300W;
      else if (database_string == "ibug")
        database_type = IBUG_300W;
      else if (database_string == "lfpw_test")
        database_type = LFPW_TEST_300W;
      else if (database_string == "lfpw_train")
        database_type = LFPW_TRAIN_300W;

      return database_type;
    }

    static ClassifierType GetClassifierType(std::string classifier_string)
    {
      ClassifierType classifier_type =
          faceAnalysis::CLASSIFIER_INIT;

      if (classifier_string == "pfq")
        classifier_type = faceAnalysis::POLY_FULL_QUADRATIC;
      else if (classifier_string == "pql")
        classifier_type = faceAnalysis::POLY_ONLY_QUADRATIC_LINEAR;
      else if (classifier_string == "pl")
        classifier_type = faceAnalysis::POLY_LINEAR;

      return classifier_type;
    }

    static ImagesToUseType GetImagesToUseType(
        std::string images_to_use_string) {
      ImagesToUseType images_to_use_type = IMAGE_TO_USE_INIT;

      if (images_to_use_string == "all")
        images_to_use_type = ALL;
      else if (images_to_use_string == "first")
        images_to_use_type = FIRST;
      else if (images_to_use_string == "random")
        images_to_use_type = RANDOM;

      return images_to_use_type;

    }
  };
}
#endif // INPUTMAPPING_H
