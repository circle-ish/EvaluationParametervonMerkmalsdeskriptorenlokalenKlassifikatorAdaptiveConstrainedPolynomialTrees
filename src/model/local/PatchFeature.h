/*
 * File:   PatchFeature.h
 * Author: ttoenige, ckrueger
 *
 */

#ifndef PATCHFEATURE_H
#define	PATCHFEATURE_H

#include <iostream>
#include <stdio.h>
#include <iterator>
#include <cstring>
#include <exception>

#include <log4cxx/logger.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "PatchFeatureUtils.h"
#include "model/Types.h"

namespace faceAnalysis {
    class PatchFeature {

    public:
        PatchFeature();
        PatchFeature(int verbosity);
        virtual ~PatchFeature();

        void Grad(
                const cv::Mat &im,
                cv::Mat &grad
                );

        void LBP(
                const cv::Mat &im,
                cv::Mat &lbp
                );

        void HOG(
                const cv::Mat &im,
                MatrixOfMats_t &blocks,
                const cv::Size cellSize,
                const cv::Size blockSize,
                const cv::Size blockOverlap,
                const int binNo,
                const bool useSign = false,
                const int offsetNo = 12,
                const bool cohog = false
                );

        /** performs MRLBP;
          * half indicates if only the upper half of the ring should be used*/
        void MultiRingLBP(
                const cv::Mat &im,
                cv::Mat &returnMat,
                const int ringNo,
                const bool half = true
                );

        /** checks wether with the current params a hog can be performed*/
        void checkHOGParameters(
                const cv::Size &im,
                cv::Size &cellNo,
                cv::Size &blockNo,
                const cv::Size &cellSize,
                const cv::Size &blockSize,
                const cv::Size &blockOverlap,
                const int binNo,
                const bool useSign
                );

        /** checks wether with the current params a cohog can be performed;
          * additional to checkHOGParameters() */
        void checkCoHOGParameters(
                const cv::Size &cellSize,
                const int offsetNo,
                int &xShift,
                int &yShift,
                int &xShiftRight
                );

        /** returns a two channel matrix (hGrad) containing the bin number the pixel
          * belongs in and the gradient magnitude*/
        void HOGGradients(
                const cv::Mat &im,
                cv::Mat &hGrad,
                const bool useSign,
                const int binNo
                );

        /** creates a grid of cells and calculates a histogramm for each;
          * cellNo given by checkHOGParameters()*/
        void HOGBinning(
                const cv::Mat &hGrad,
                MatrixOfMats_t &cellBins,
                const cv::Size cellSize,
                const cv::Size cellNo,
                const int binNo
                );

        /** creates a grid of cells and calculates a histogramm for each;
         * the shifts are calculated by checkCoHOGParameters()*/
        void CoHOGBinning(
                const cv::Mat& hGrad,
                MatrixOfMats_t &cellBins,
                const cv::Size cellSize,
                const cv::Size cellNo,
                const int binNo,
                const int offsetNo,
                const int xShift,
                const int yShift,
                const int xShiftRight
                );

        /** unites multiple cells to blocks*/
        void HOGComputeBlocks(
                MatrixOfMats_t &blocks,
                const MatrixOfMats_t &cells,
                const cv::Size blockNo,
                const cv::Size blockSize,
                const cv::Size blockOverlap
                );

        int verbosity;
    };
}
#endif	/* PATCHFEATURE_H */

