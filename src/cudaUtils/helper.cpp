#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <string>

cv::Mat imageInput;
cv::Mat imageOutput;
cv::Mat Ainput;

unsigned char *d_inputImage__;
double *d_answerImage__;
double *d_AMat__;





//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device

void preProcess(
        unsigned char **inputImage,
        double **answerImage,
        unsigned char **d_inputImage,
        double **d_answerImage,
        cv::Mat image,
        double **AMat,
        double **d_AMat,
        cv::Mat A
        ) {
    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    int faceWidth = image.cols;
    double scaleFactor = faceWidth / (double) faceWidth;
    cv::resize(image, image, cv::Size(faceWidth, scaleFactor * image.rows));

    //    cv::cvtColor(image, imageInput, CV_BGR2RGBA);
    imageInput = image;

    //allocate memory for the output
    imageOutput.create(image.rows, image.cols, CV_64F);


    Ainput = A;

    //This shouldn't ever happen given the way the images are created
    //at least based upon my limited understanding of OpenCV, but better to check
    if (!imageInput.isContinuous() || !imageOutput.isContinuous()) {
        std::cerr << "Images aren't continuous!! Exiting." << std::endl;
        exit(1);
    }


    *inputImage = imageInput.ptr<unsigned char>(0);
    *answerImage = imageOutput.ptr<double>(0);
    *AMat = Ainput.ptr<double>(0);

    const size_t numPixels = imageInput.rows * imageInput.cols;
    //allocate memory on the device for both input and output
    checkCudaErrors(cudaMalloc(d_inputImage, sizeof (unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_answerImage, sizeof (double) * numPixels));
    checkCudaErrors(cudaMalloc(d_AMat, sizeof (double) * A.rows * A.cols));
    checkCudaErrors(cudaMemset(*d_answerImage, 0, numPixels * sizeof (double))); //make sure no memory is left laying around

    //copy input array to the GPU
    checkCudaErrors(cudaMemcpy(*d_inputImage, *inputImage, sizeof (unsigned char) * numPixels, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_AMat, *AMat, sizeof (double) * A.rows * A.cols, cudaMemcpyHostToDevice));

    d_inputImage__ = *d_inputImage;
    d_answerImage__ = *d_answerImage;
    d_AMat__ = *d_AMat;
}

void postProcess() {
    const int numPixels = imageInput.rows * imageInput.cols;
    //copy the output back to the host
    checkCudaErrors(cudaMemcpy(imageOutput.ptr<double>(0), d_answerImage__, sizeof (double) * numPixels, cudaMemcpyDeviceToHost));


    //    std::cout << imageOutput << std::endl;

    //output the image
    //      cv::imwrite(output_file.c_str(), imageGrey);
    //    cv::imshow("output", imageOutput);
    //    cv::waitKey(0);

    //cleanup
    cudaFree(d_inputImage__);
    cudaFree(d_answerImage__);
    cudaFree(d_AMat__);
}
