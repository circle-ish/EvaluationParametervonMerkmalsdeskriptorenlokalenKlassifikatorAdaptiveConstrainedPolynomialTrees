#ifndef PATCHFEATUREUTILS_H
#define PATCHFEATUREUTILS_H

#include <iostream>
#include <stdio.h>
#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "model/Types.h"

namespace faceAnalysis {
    class PatchFeatureUtils {
    public:
        /** creates a single matrix from the histograms of a grid of cells or blocks;
          * shape determines if the output is a matrix or
          * an vertical or horizontal vector */
        static void BinPatching(
                const MatrixOfMats_t& src,
                cv::Mat &output,
                const BinPatchingShapeType shape = SHAPE_VERTICAL,
                const bool rotate = false
                );
        /** extracts a submatrix */
        static void getMatrixOfMatsRoi(
                const faceAnalysis::MatrixOfMats_t &src,
                faceAnalysis::MatrixOfMats_t &dst,
                const cv::Rect &roi
                );
        /** prints a single column to std::out;
          * default is a column in the middle */
        static void MatPrettyPrint(
                const cv::Mat &src,
                int col = -1
                );
        static void rotateMatrix(
                const cv::Mat &src,
                cv::Mat &dst
                );
        /** converts a matrix to CV_8U */
        static void convertMatrix(
                const cv::Mat &src,
                cv::Mat &dst
                );
        /** displays any matrix via cv::imshow */
        static void showMatrix(
                const cv::Mat &src,
                const std::string message,
                const int time = 0
                );

        static void calculateCellBlockNumber(
                cv::Size &cellNo,
                cv::Size &blockNo,
                const cv::Size &imSize,
                const cv::Size &cellSize,
                const cv::Size &blockSize,
                const cv::Size &blockOverlap
                );
        static int nChoosek(
                const int n,
                int k
                );

        /** enlarges the given matrix by factor */
        template <typename T>
        static void zoomMatrix(
                const cv::Mat &src,
                cv::Mat &dst,
                const int factor
                ) {

            const int h = src.rows;
            const int w = src.cols;
            const int dstH = h * factor;
            const int dstW = w * factor;
            dst.create(dstH + factor, dstW + factor, src.type());
            cv::Mat roi(dst(cv::Rect(0, 0, factor, factor)));
            cv::MatConstIterator_<T> srcPtr = src.begin<T>();

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    ++srcPtr;
                    roi.setTo(cv::Scalar(*srcPtr));
                    roi.adjustROI(0, 0, -factor, factor);
                }
                roi.adjustROI(-factor, factor, dstW, -dstW);
            }

            dst.adjustROI(0, -factor, 0, -factor);
        }

        static void freeMatrixOfMats(
                MatrixOfMats_t &mat
                );

    };
}

#endif // PATCHFEATUREUTILS_H
