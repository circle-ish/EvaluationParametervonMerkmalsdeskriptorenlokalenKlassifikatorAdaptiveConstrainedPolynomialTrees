#include "PatchFeatureUtils.h"

using namespace faceAnalysis;

void PatchFeatureUtils::MatPrettyPrint(
        const cv::Mat& src,
        int col
        ) {

    std::stringstream s;
    if (src.cols == 1) {
        s << "[";
        if (src.channels() == 2) {
            if (src.type() == GRAD_CV_TYPE) {
                for (
                     cv::MatConstIterator_<GRAD_VEC_TYPE> it = src.begin<GRAD_VEC_TYPE>();
                     it != src.end<GRAD_VEC_TYPE>();
                     ++it
                     ) {

                    s << (*it) << " ";
                }
            } else {
                s << "nothing matched";
            }
        } else if (src.channels() == 1) {
            if (src.type() == SRC_CV_TYPE) {
                for (
                     cv::MatConstIterator_<SRC_TYPE> it = src.begin<SRC_TYPE>();
                     it != src.end<SRC_TYPE>();
                     ++it
                     ) {

                    s << (*it) << " ";
                }
            } else if (src.type() == BIN_CV_TYPE) {
                for (
                     cv::MatConstIterator_<BIN_TYPE> it = src.begin<BIN_TYPE>();
                     it != src.end<BIN_TYPE>();
                     ++it
                     ) {

                    s << (*it) << " ";
                }
            } else if (src.type() == MR_CV_TYPE) {
                for (
                     cv::MatConstIterator_<MR_TYPE> it = src.begin<MR_TYPE>();
                     it != src.end<MR_TYPE>();
                     ++it
                     ) {

                    s << (*it) << " ";
                }
            } else {
                s << "nothing matched";
            }
        }

        s << "];";
        std::cout << s.str() << std::endl;

    } else {
        if (col == -1) {
            col = src.cols / 2;
        }
        PatchFeatureUtils::MatPrettyPrint(src.col(col));
    }
}

void PatchFeatureUtils::BinPatching(
        const MatrixOfMats_t &src,
        cv::Mat &dst,
        const BinPatchingShapeType shape,
        const bool rotate
        ) {

    const cv::Mat sample = src[0][0];
    const int h = src.size();
    const int w = src[0].size();
    const int sh = sample.rows;
    const int sw = sample.cols;
    const int fullW = w * sw;
    const int fullH = h * sh;

    if (shape == SHAPE_QUADRATIC) {
        cv::Mat output(cv::Size(fullW + sw, fullH + sh), sample.type());
        cv::Mat roi(output(cv::Rect(0, 0, sw, sh)));

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                src[i][j].copyTo(roi);
                roi.adjustROI(0, 0, -sw, sw);
            }
            roi.adjustROI(-sh, sh, fullW, -fullW);
         }

        output.adjustROI(0, -sh, 0, -sw);
        output.copyTo(dst);
    } else if (shape == SHAPE_VERTICAL) {
        const int vert = sw * sh;
        cv::Mat output(cv::Size(1, fullW * fullH), sample.type());
        cv::Mat roi(output(cv::Rect(0, 0, 1, vert)));

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cv::Mat tmp = src[i][j];
                tmp.reshape(0, tmp.total()).copyTo(roi);
                roi.adjustROI(-vert, vert, 0, 0);
            }
         }
        output.copyTo(dst);
    } else if (shape == SHAPE_HORIZONTAL) {
        const int horiz = sw * sh;
        cv::Mat output(cv::Size(fullW * fullH, 1), sample.type());
        cv::Mat roi(output(cv::Rect(0, 0, horiz, 1)));

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cv::Mat tmp = src[i][j];
                tmp.reshape(0, 1).copyTo(roi);
                roi.adjustROI(0, 0, -horiz, horiz);
            }
         }
        output.copyTo(dst);
    }

    if (rotate) {
        PatchFeatureUtils::rotateMatrix(dst, dst);
    }
}

void PatchFeatureUtils::getMatrixOfMatsRoi(
        const faceAnalysis::MatrixOfMats_t &src,
        faceAnalysis::MatrixOfMats_t &dst,
        const cv::Rect &roi
        ) {

    dst.reserve(roi.height);
    faceAnalysis::MatrixOfMats_t::const_iterator itY = src.begin() + roi.y;
    for (int y = 0; y < roi.height; ++y) {

        std::vector<cv::Mat> vec;
        vec.reserve(roi.width);
        std::vector<cv::Mat>::const_iterator itX = (*itY).begin() + roi.x;
        for (int x = 0; x < roi.width; ++x) {
            vec.push_back(*itX);
            ++itX;
        }

        dst.push_back(vec);
        ++itY;
    }
}

void PatchFeatureUtils:: showMatrix(
        const cv::Mat &src,
        const std::string message,
        const int time
        ) {

    if (src.channels() > 1) {
        cv::Mat split[src.channels()];
        cv::split(src, split);

        for (int i = 0; i < src.channels(); i++) {
            cv::Mat output;
            PatchFeatureUtils::convertMatrix(split[i], output);

            std::stringstream s;
            s << message << ", channel " << i;
            cv::imshow(s.str(), output);
        }
    } else {
        cv::Mat output;
        PatchFeatureUtils::convertMatrix(src, output);

        cv::imshow(message, output);
    }

    cv::waitKey(time);
}

void PatchFeatureUtils::rotateMatrix(
        const cv::Mat &src,
        cv::Mat &dst
        ) {

    cv::Point2f pt(src.rows/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, 270, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.rows, src.cols));
}

void PatchFeatureUtils::convertMatrix(
        const cv::Mat &src,
        cv::Mat &dst
        ) {

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
    src.convertTo(
                dst,
                CV_8U,
                255.0 / (maxVal - minVal),
                -255.0 * minVal / (maxVal - minVal)
//                -minVal
                );
}

void PatchFeatureUtils::calculateCellBlockNumber(
        cv::Size &cellNo,
        cv::Size &blockNo,
        const cv::Size &imSize,
        const cv::Size &cellSize,
        const cv::Size &blockSize,
        const cv::Size &blockOverlap
        ) {

    const int cellSizeDivW = (imSize.width / cellSize.width);
    const int cellSizeDivH = (imSize.height / cellSize.height);
    cellNo = cv::Size(cellSizeDivH, cellSizeDivW);

    /**
      * calculating the number of blocks
      * exemplarily for a row:
      * #Blocks = (#cells - #OfCellsInABlock) / (#OfCellsInABlock - #OfCellsThatOverlap) + 1
      *
      * these parameter are valid only, if the remainder of the division is 0
      * otherwise it would be a fraction number of blocks
      */
    const std::div_t h = std::div((cellNo.height - blockSize.height),
                                        (blockSize.height - blockOverlap.height));
    const std::div_t w = std::div((cellNo.width - blockSize.width),
                                        (blockSize.width - blockOverlap.width));
    blockNo = cv::Size(w.quot + 1, h.quot + 1);
}

int PatchFeatureUtils::nChoosek(
        const int n,
        int k
        ) {

    if (0 == k || n == k) {
        return 1;
    }
    if (k > n) {
        return 0;
    }
    if (k > (n - k)) {
        k = n - k;
    }
    if (1 == k) {
        return n;
    }

    unsigned long res = 1;
    for (int i = 1; i <= k; ++i) {
        res *= (n - (k - i));
//        if (res < 0) {
//            return -1; /* Overflow */
//        }
        res /= i;
    }
    return res;
}

void PatchFeatureUtils::freeMatrixOfMats(MatrixOfMats_t &mat) {
    while (!mat.empty()) {
        std::vector<cv::Mat> sndD = mat.back();
        while (!sndD.empty()) {
            sndD.back().release();
            sndD.pop_back();
        }
        mat.pop_back();
    }
}
