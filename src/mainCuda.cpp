#include "utils.h"
#include "cudaUtils/GpuTimer.h"
#include "CPT.h"
#include "Types.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>

#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/helpers/exception.h>


using namespace std;
using namespace faceAnalysis;
using namespace log4cxx;
using namespace log4cxx::helpers;
using namespace boost::filesystem;


size_t numRows(); //return # of rows in the image
size_t numCols(); //return # of cols in the image

void preProcess(
        unsigned char **h_inputImage,
        double **h_answerImage,
        unsigned char **d_inputImage,
        double **d_answerImage,
        cv::Mat image,
        double **h_AMat,
        double **d_AMat,
        cv::Mat A
        );

void postProcess();

void startCalcAnswerMap(
        unsigned char* const d_inputImage,
        double* const d_answerImage,
        size_t numXInput,
        size_t numYInput,
        size_t numZInput,
        double* const d_AMat,
        int AMatLength
        );

LoggerPtr logger(Logger::getLogger("CPT.mainCuda"));

int main(int argc, char **argv) {
    unsigned char *h_inputImage, *d_inputImage;
    double *h_answerImage, *d_answerImage;
    double *h_AMat, *d_AMat;

    std::string input_file;
    if (argc == 3) {
        input_file = std::string(argv[1]);
    } else {
        std::cerr << "Usage: ./hw input_file output_file" << std::endl;
        exit(1);
    }

    string clmDataPath = "/homes/ttoenige/NetBeansProjects/FaceAnalysis/CPT/trunk/config/";
    string cascade = OPEN_CV_SHARE"/haarcascades/haarcascade_frontalface_alt2.xml";
    string outputFolderPath = OUTPUT_FOLDER;

    stringstream loggerPath;
    loggerPath << clmDataPath << "logger.cfg";
    if (exists(loggerPath.str())) {
        PropertyConfigurator::configure(loggerPath.str());
        LOG4CXX_INFO(logger, "Using logger config file " << loggerPath.str());
    } else {
        BasicConfigurator::configure();
        LOG4CXX_INFO(logger, "Using basic logger");
    }

    CPT cpt(cascade, POLY_ONLY_QUADRATIC_LINEAR);
    cpt.load(outputFolderPath);






    cv::Mat A_test = cpt.getLocalPatches()[0].getA();

    cv::Mat image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(image, image, CV_RGB2GRAY);

    cpt.initFeatureMap(image);

    //load the image and give us our input and output pointers
    preProcess(&h_inputImage, &h_answerImage, &d_inputImage, &d_answerImage, image, &h_AMat, &d_AMat, A_test);

    GpuTimer timer;
    timer.Start();
    startCalcAnswerMap(d_inputImage, d_answerImage, image.rows, image.cols, -1, d_AMat, A_test.rows);
    timer.Stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    printf("\n");

    printf("%f msecs.\n", timer.Elapsed());


    //check results and output the grey image
    postProcess();

    cpt.freeFeatureMap();

    return 0;
}
