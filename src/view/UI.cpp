#include "UI.h"

#include <iostream>

#include <opencv2/highgui/highgui.hpp>

const std::string faceAnalysis::UI::windowName = "result";

faceAnalysis::UI::UI()
{
  cv::namedWindow(windowName, 1);
}

void faceAnalysis::UI::setCPT(CPTPtr cpt)
{
  int localWeight = cpt->weight_local() * 10;
  int globalWeight = cpt->weight_global() * 10;

  cv::createTrackbar("local", "", &localWeight, 200, setLocalWeightChange,
                     cpt.get());
  cv::createTrackbar("global", "", &globalWeight, 200, setGlobalWeightChange,
                     cpt.get());
  cv::createButton("show local details", showLocalDetails, cpt.get(),
                   CV_PUSH_BUTTON, 0);

  cv::createButton("redetect", redetect, cpt.get(), CV_PUSH_BUTTON, 0);
  cv::createButton("reinit", redetectinit, cpt.get(), CV_PUSH_BUTTON, 0);
}


void faceAnalysis::UI::setLocalWeightChange(int localWeight, void* userdata)
{
  UNUSED(localWeight);
  UNUSED(userdata);
}


void faceAnalysis::UI::setGlobalWeightChange(int globalWeight, void* userdata)
{
  UNUSED(globalWeight);
  UNUSED(userdata);
}

void faceAnalysis::UI::showLocalDetails(int state, void* userdata)
{
  UNUSED(state);
  ((CPT*) userdata)->ShowLocalDetails();
}

void faceAnalysis::UI::setLocalWeight(int state, void* userdata)
{
  UNUSED(state);
  double lW = cv::getTrackbarPos("local", windowName) / (double) 10;
  std::cout << "set local weight to " << lW << std::endl;
  ((CPT*) userdata)->set_weight_local(lW);
}

void faceAnalysis::UI::setGlobalWeight(int state, void* userdata)
{
  UNUSED(state);
  double gW = cv::getTrackbarPos("global", windowName) / (double) 10;
  std::cout << "set global weight to " << gW << std::endl;
  ((CPT*) userdata)->set_weight_global(gW);
}

void faceAnalysis::UI::redetect(int state, void* userdata)
{
  UNUSED(state);
  double lW = cv::getTrackbarPos("local", windowName) / (double) 10;
  std::cout << "set local weight to " << lW << std::endl;
  double gW = cv::getTrackbarPos("global", windowName) / (double) 10;
  std::cout << "set global weight to " << gW << std::endl;
  ((CPT*) userdata)->set_weight_local(lW);
  ((CPT*) userdata)->set_weight_global(gW);
  ((CPT*) userdata)->RedetectFeatures(false);
}

void faceAnalysis::UI::redetectinit(int state, void* userdata)
{
  UNUSED(state);
  double lW = cv::getTrackbarPos("local", windowName) / (double) 10;
  std::cout << "set local weight to " << lW << std::endl;
  double gW = cv::getTrackbarPos("global", windowName) / (double) 10;
  std::cout << "set global weight to " << gW << std::endl;
  ((CPT*) userdata)->set_weight_local(lW);
  ((CPT*) userdata)->set_weight_global(gW);
  ((CPT*) userdata)->RedetectFeatures(true);

}
