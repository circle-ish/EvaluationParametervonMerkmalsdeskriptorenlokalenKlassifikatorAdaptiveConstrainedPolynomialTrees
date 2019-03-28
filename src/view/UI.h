#ifndef UI_H
#define	UI_H

#include "model/trackers/CPT.h"

#include <log4cxx/logger.h>

namespace faceAnalysis {
  class UI   {
    static const std::string windowName;
  public:
    UI();
    ~UI() {}

    void setCPT(CPTPtr cpt);

  private:
    static void setLocalWeightChange(int localWeight, void* userdata);
    static void setGlobalWeightChange(int globalWeight, void* userdata);
    static void showLocalDetails(int state, void* userdata);
    static void setLocalWeight(int state, void* userdata);
    static void setGlobalWeight(int state, void* userdata);
    static void redetect(int state, void* userdata);
    static void redetectinit(int state, void* userdata);
  };
}

#endif	/* UI_H */

