/**
  * File:       PatchFeature.cpp
  * Author:     ttoenige, ckrueger
  * Purpose:    Extracts different features from local patches of an image.
  *
  */

#include "PatchFeature.h"
//heller = kleiner; dunkler = groesser
#define SGN(x) (x<0) ? 0:1
#define SGN0(x) (x<-3) ? 0:1
#define SGN1(x) (x<8) ? 0:1
#define SGN2(x) (x<-2) ? 0:1
#define SGNp2(x) (x<-10) ? 0:1
#define SGN3(x) (x<-2) ? 0:1
#define SGNp3(x) (x<-30) ? 0:1

using namespace std;
using namespace faceAnalysis;

static log4cxx::LoggerPtr logger;

PatchFeature::PatchFeature() {
    logger = log4cxx::Logger::getLogger("PatchFeature");
    verbosity = 0;
}

PatchFeature::PatchFeature(int verbosity) {
    logger = log4cxx::Logger::getLogger("PatchFeature");
    PatchFeature::verbosity = verbosity;
}

PatchFeature::~PatchFeature() {
}


// same as CLM

void PatchFeature::Grad(
        const cv::Mat &im,
        cv::Mat &grad
        ) {

    //    cout << "Grad" << endl;
    //    assert((im.rows == grad.rows) && (im.cols == grad.cols));
    //    assert((im.type() == CV_32F) && (grad.type() == CV_32F));
    int x, y, h = im.rows, w = im.cols;
    double vx, vy;
    grad.create(h - 2, w - 2, CV_64F);
    grad.setTo(cv::Scalar(0));
    cv::MatIterator_<double> gp = grad.begin<double>();
    cv::MatConstIterator_<double> px1 = im.begin<double>() + w + 2;
    cv::MatConstIterator_<double> px2 = im.begin<double>() + w;
    cv::MatConstIterator_<double> py1 = im.begin<double>() + 2 * w + 1;
    cv::MatConstIterator_<double> py2 = im.begin<double>() + 1;
    for (y = 0; y < grad.rows; y++) {
        for (x = 0; x < grad.cols; x++) {
            vx = *px1++ - *px2++;
            vy = *py1++ - *py2++;
            *gp++ = vx * vx + vy * vy;
        }
        px1 += 2;
        px2 += 2;
        py1 += 2;
        py2 += 2;
    }
    return;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
///                 LBP - Variants                                                      ///////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// same as CLM
void PatchFeature::LBP(
        const cv::Mat &im,
        cv::Mat &lbp
        ) {

    //    cout << "LBP" << endl;
    int x, y, h = im.rows, w = im.cols;
    double v[9];
    lbp.create(h - 2, w - 2, CV_64F);
    lbp.setTo(cv::Scalar(0));
    cv::MatIterator_<double> lp = lbp.begin<double>();
    cv::MatConstIterator_<double> p1 = im.begin<double>();
    cv::MatConstIterator_<double> p2 = im.begin<double>() + w;
    cv::MatConstIterator_<double> p3 = im.begin<double>() + w * 2;
    for (y = 0; y < lbp.rows; y++) {
        for (x = 0; x < lbp.cols; x++) {
            v[4] = *p2++;
            v[0] = *p2++;
            v[5] = *p2;
            v[1] = *p1++;
            v[2] = *p1++;
            v[3] = *p1;
            v[6] = *p3++;
            v[7] = *p3++;
            v[8] = *p3;
            *lp++ = (SGN(v[0] - v[1]) * 2) + (SGN(v[0] - v[2]) * 4) + (SGN(v[0] - v[3])
                    * 8) +(SGN(v[0] - v[4]) * 16) + (SGN(v[0] - v[5])
                    * 32) + (SGN(v[0] - v[6]) * 64) + (SGN(v[0] - v[7])
                    * 128) + (SGN(v[0] - v[8]) * 256);

            p1--;
            p2--;
            p3--;
        }
        p1 += 2;
        p2 += 2;
        p3 += 2;
    }
    return;
}

void PatchFeature::MultiRingLBP(
        const cv::Mat &im,
        cv::Mat &returnMat,
        const int ringNo,
        const bool half
        ) {

    const int w = im.cols;
    const int h = im.rows;

    /** set mean to zero */
    cv::Mat imConv = im - mean(im);
//    unsigned short iPatternNo = std::pow(2, ringNo);
//    unsigned short ePatternNo = std::pow(2, PatchFeatureUtils::nChoosek(ringNo + 1, 2));

    returnMat.create(h - 2 * ringNo, w - 2 * ringNo, MR_CV_TYPE);
    returnMat.setTo(cv::Scalar(0, 0, 0, 0));

    cv::Mat mat(returnMat.rows, returnMat.cols, CV_64F);
    cv::Mat onesMNo(returnMat.rows, returnMat.cols, CV_16UC3, cv::Scalar(0));            //hardcoded
    cv::Mat onesPNo(returnMat.rows, returnMat.cols, CV_16UC3, cv::Scalar(0));            //hardcoded
    cv::Mat sumFor3Rings(returnMat.rows, returnMat.cols, CV_16UC4, cv::Scalar(0));       //hardcoded
    
//    cv::MatIterator_<double> matt = mat.begin<double>();
    cv::MatConstIterator_<SRC_TYPE> p = imConv.begin<SRC_TYPE>() + ringNo * w + ringNo;
    cv::MatIterator_<MR_VEC_TYPE> ret = returnMat.begin<MR_VEC_TYPE>();
    cv::MatIterator_<cv::Vec3w> onesM = onesMNo.begin<cv::Vec3w>();             //hardcoded
    cv::MatIterator_<cv::Vec3w> onesP = onesPNo.begin<cv::Vec3w>();             //hardcoded
    cv::MatIterator_<cv::Vec4w> sum = sumFor3Rings.begin<cv::Vec4w>();          //hardcoded

    /**
      variables:
        ret[0] erlbp- = relationship between local rings
        ret[1] erlbp+ = relationship between two rings and the global pixel value mean
        ret[2] irlbp- = relationship between local points and their centerpoint
        ret[3] irlbp+ = relationship between a local point, his centerpoint and the global pixel value mean

        extra-ring lbp:
            sum[i] = for one ring the sum of his pixels

        intra-ring lbp:
            onesM[i] = counts the ones in a ring returned by SGN for irlbp-
            onesP[i] = counts the ones in a ring returned by SGN for irlbp+

        auxilaries
            tmp = value of the compared pixel
            offset = address difference to the compared ring pixel

     */

    /** for every pixel, for every ring do sth*/
    for (int y = ringNo; y < h - ringNo; y++) {
        for (int x = ringNo; x < w - ringNo; x++) {
            (*sum)[0] = *p;
            for (int i = 1; i < ringNo + 1; i++) {

                /** left upper vertical*/
                int help = ringNo + i - 1;
                int offset = -i * (w + 1);
                for (int ry = -i + 1; ry < 1; ry++) {
                    offset += w;
                    const int tmp = *(p + offset);
                    (*onesM)[i] += SGNp2(tmp - *p);
                    (*onesP)[i] += SGNp3(tmp + *p);

                    (*sum)[i] += tmp;

                    if (!half && help < x && help - ry - 1 < y) {
                        (*(onesM + offset))[i] += SGNp2(*p - tmp);
                        (*(onesP + offset))[i] += SGNp3(*p + tmp);

                        (*(sum + offset))[i] += tmp;
                    }
                }
                /** upper left horizontal */
                offset = -i * (w + 1) - 1;
                for (int rx = -i; rx < 1; rx++) {
                    offset++;
                    const int tmp = *(p + offset);
                    (*onesM)[i] += SGNp2(tmp - *p);
                    (*onesP)[i] += SGNp3(tmp + *p);

                    (*sum)[i] += tmp;

                    if (!half && ringNo - rx - 1 < x && help < y) {
                        (*(onesM + offset))[i] += SGNp2(*p - tmp);
                        (*(onesP + offset))[i] += SGNp3(*p + tmp);

                        (*(sum + offset))[i] += tmp;
                    }
                }
                /** upper right horizontal */
                for (int rx = 1; rx < i + 1; rx++) {
                    offset++;
                    const int tmp = *(p + offset);
                    (*onesM)[i] += SGNp2(tmp - *p);
                    (*onesP)[i] += SGNp3(tmp + *p);

                    (*sum)[i] += tmp;

                    if (!half && x + rx < w - ringNo && help < y) {
                        (*(onesM + offset))[i] += SGNp2(*p - tmp);
                        (*(onesP + offset))[i] += SGNp3(*p + tmp);

                        (*(sum + offset))[i] += tmp;
                    }
                }
                /** right upper vertical */
                offset = -i * (w - 1);
                for (int ry = -i + 1; ry < 0; ry++) {
                    offset += w;
                    const int tmp = *(p + offset);
                    (*onesM)[i] += SGNp2(tmp - *p);
                    (*onesP)[i] += SGNp3(tmp + *p);

                    (*sum)[i] += tmp;

                    if (!half && x + help + 1 < w && help - ry - 1 < y) {
                        (*(onesM + offset))[i] += SGNp2(*p - tmp);
                        (*(onesP + offset))[i] += SGNp3(*p + tmp);

                        (*(sum + offset))[i] += tmp;
                    }
                }
            }

            p++;
            onesM++;
            onesP++;
            sum++;
//            matt++;
        }

        p += 2 * ringNo;
    }

    ret = returnMat.begin<MR_VEC_TYPE>();
    onesM = onesMNo.begin<cv::Vec3w>();             //hardcoded
    onesP = onesPNo.begin<cv::Vec3w>();             //hardcoded
    sum = sumFor3Rings.begin<cv::Vec4w>();          //hardcoded
    for (int y = 0; y < returnMat.rows; ++y) {
        for (int x = 0; x < returnMat.cols; ++x) {
            for (int k = 0; k < ringNo; k++) {
                for (int j = k + 1; j < ringNo + 1; j++) {
                    /** erlbp;
                     * power is 0, 1, 2, 3...*/

                    int power = (2 * ringNo - k - 1) * k / 2 + (j - 1);
                    double mean1 = (*(sum))[j] / (double)(j * 8);
                    double mean2 = (*(sum))[k] / (double)std::pow(k * 8, std::min(1, k));

                    (*(ret))[0] += SGN0(mean2 - mean1) * power;
                    (*(ret))[1] += SGN1(mean2 + mean1) * power;
                }

                /** irlbp
                  * #ones - #minuses = #ones - (#total - #ones) = 2*#ones - #total    and
                  * sgn(2*#ones - 8*i) = sgn(#ones - 4*i)*/
                (*(ret))[2] += SGN2((*(onesM))[k] - (k * 4)) * std::pow(2, k);
                (*(ret))[3] += SGN3((*(onesP))[k] - (k * 4)) * std::pow(2, k);
            }

            ++onesM;
            ++onesP;
            ++sum;
            ++ret;
        }
    }

//    mat = split[0]/* * ePatternNo * iPatternNo * iPatternNo*/
//            + split[1] /** iPatternNo * iPatternNo*/
//            + split[2] //* iPatternNo
//            + split[3];
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///                 HOG - Variants                                                      ///////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void PatchFeature::HOG(
        const cv::Mat &im,
        MatrixOfMats_t &blocks,
        const cv::Size cellSize,
        const cv::Size blockSize,
        const cv::Size blockOverlap,
        const int binNo,
        const bool useSign,
        const int offsetNo,
        const bool cohog
        ) {

    if (verbosity > 0) {
        LOG4CXX_INFO(logger, "called hog()");
    }
    if (verbosity > 4) {
        PatchFeatureUtils::showMatrix(im, "obama, src");
    }

    cv::Size cellNo, blockNo;
    int xShift, yShift, xShiftRight;
    const cv::Size imgSize = im.size();

    try {
        PatchFeature::checkHOGParameters(
                    imgSize,
                    cellNo,
                    blockNo,
                    cellSize,
                    blockSize,
                    blockOverlap,
                    binNo,
                    useSign
                    );
        if (cohog) {
            PatchFeature::checkCoHOGParameters(
                        cellSize,
                        offsetNo,
                        xShift,
                        yShift,
                        xShiftRight
                        );
        }
    } catch (std::string message) {
        throw message;
    }

    /** compute gradients */
    cv::Mat binMag(im.rows, im.cols, BINMAG_CV_TYPE, cv::Scalar(0));
    cv::Mat borderIm;
    cv::copyMakeBorder(im, borderIm, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

    try {
        PatchFeature::HOGGradients(
                    borderIm,
                    binMag,
                    useSign,
                    binNo
                    );
    } catch (std::string message) {
        throw message;
    }

    if (verbosity > 2) {
        PatchFeatureUtils::showMatrix(binMag, "after HOGGradients return: binMag");
    }

    PatchFeatureUtils::calculateCellBlockNumber(
                cellNo,
                blockNo,
                im.size(),
                cellSize,
                blockSize,
                blockOverlap
                );
    /** bin gradients; histogram for cells*/
    MatrixOfMats_t cellBins;
    if (cohog) {
        PatchFeature::CoHOGBinning(
                    binMag,
                    cellBins,
                    cellSize,
                    cellNo,
                    binNo,
                    offsetNo,
                    xShift,
                    yShift,
                    xShiftRight
                    );
    } else {
        PatchFeature::HOGBinning(
                    binMag,
                    cellBins,
                    cellSize,
                    cellNo,
                    binNo
                    );
    }

    if (verbosity > 4) {
        std::cout << "binSize=" << cellBins.size()
                << " -- sample histogram:";
        PatchFeatureUtils::MatPrettyPrint(cellBins[0][0], 0);
    }
    if (verbosity > 2) {
        cv::Mat output;
        PatchFeatureUtils::BinPatching(cellBins, output, faceAnalysis::SHAPE_QUADRATIC);
        PatchFeatureUtils::showMatrix(output, "all histograms patched together");
    }

    /** create vectors of histograms for blocks*/
    if (verbosity > 4) {
        std::cout << "resizing MatrixOfMats_t vector; height=" << blockNo.height << " width=" << blockNo.width << std::endl;
    }

    try {
        PatchFeature::HOGComputeBlocks(
                blocks,
                cellBins,
                blockNo,
                blockSize,
                blockOverlap
                );
    } catch (std::string message) {
        throw message;
    }
}

void PatchFeature::checkHOGParameters(
        const cv::Size &im,
        cv::Size &cellNo,
        cv::Size &blockNo,
        const cv::Size &cellSize,
        const cv::Size &blockSize,
        const cv::Size &blockOverlap,
        const int binNo,
        const bool useSign
        ) {

    if (verbosity > 0) {
        LOG4CXX_INFO(logger, "called checkHOGParameters()"
                     << "\n\t imagecv::Size=" << im
                     << " cellSize=" << cellSize << " blockSize=" << blockSize
                     << " blockOverlap=" << blockOverlap << " binNo=" << binNo);
    }

    /** check obvious stuff*/
    if (cellSize.width < 1 || cellSize.height < 1) {
        std::stringstream error;
        error << "checkHOGParameters(): cellSize has to be a positiv number";
        LOG4CXX_ERROR(logger, error.str());
        throw(error.str());
    }
    if ((blockSize.height - blockOverlap.height < 1) || (blockSize.width - blockOverlap.width < 1)) {
        std::stringstream error;
        error << "checkHOGParameters(): blockSize has to be bigger than blockOverlap";
        LOG4CXX_ERROR(logger, error.str());
        throw(error.str());
    }
    if (blockOverlap.width < 0 || blockOverlap.height < 0) {
        std::stringstream error;
        error << "checkHOGParameters(): blockOverlap has to be a non-negativ number";
        LOG4CXX_ERROR(logger, error.str());
        throw(error.str());
    }
    if (binNo < 2) {
        std::stringstream error;
        error << "checkHOGParameters(): binNo has to be bigger than 1";
        LOG4CXX_ERROR(logger, error.str());
        throw(error.str());
    }
    if (useSign) {
        const div_t binSize = std::div(360, binNo);
        if (binSize.rem != 0) {
            std::stringstream error;
            error << "checkHOGParameters(): binNo for 360 degree allows no clean partitioning";
            LOG4CXX_ERROR(logger, error.str());
            throw(error.str());
        }
    } else {
        const div_t binSize = std::div(180, binNo);
        if (binSize.rem != 0) {
            std::stringstream error;
            error << "checkHOGParameters(): binNo for 180 degree allows no clean partitioning";
            LOG4CXX_ERROR(logger, error.str());
            throw(error.str());
        }
    }

//    /** check and calculate image / cellsize = #cellNo proportion*/
//    const int cellSizeDivW = (im.width / cellSize.width);
//    const int cellSizeModW = (im.width % cellSize.width);
//    if (cellSizeModW != 0) {
//        std::stringstream error;
//        error << "checkHOGParameters(): imagewidth=" << im.width
//              << " is not divisible by cellwidth=" << cellSize.width;
//        LOG4CXX_ERROR(logger, error.str());
//        throw(error.str());
//    }

//    const int cellSizeDivH = (im.height / cellSize.height);
//    const int cellSizeModH = (im.height % cellSize.height);
//    if (cellSizeModH != 0) {
//        std::stringstream error;
//        error << "checkHOGParameters(): imageheight   =" << im.height
//              << " is not divisible by cellheight=" << cellSize.height;
//        LOG4CXX_ERROR(logger, error.str());
//        throw(error.str());
//    }

//    /** no more cells in a block than cells at all */
//    cellNo = cv::Size(cellSizeDivH, cellSizeDivW);
//    if (cellNo.width < blockSize.width || cellNo.height < blockSize.height) {
//        std::stringstream error;
//        error << "checkHOGParameters(): blockSize(" << blockSize.height << ", "
//              << blockSize.width << ") is bigger than the overall cellNo("
//              << cellNo.height << ", " << cellNo.width << ")";
//        LOG4CXX_ERROR(logger, error.str());
//        throw(error.str());
//    }

//    /** for an explanation on blockNo see PatchFeatureUtils::calculateCellBlockNumber() */
//    const std::div_t h = std::div((cellNo.height - blockSize.height),
//                                              (blockSize.height - blockOverlap.height));
//    if (h.rem != 0) {
//        std::stringstream error;
//        error << "checkHOGParameters(): vertical block number(" << blockNo.height
//              << ") does not fit; cellnumber=" << cellNo.height
//              << " blockOverlap=" << blockOverlap.height;
//        LOG4CXX_ERROR(logger, error.str());
//        throw(error.str());
//    }

//    const std::div_t w = std::div((cellNo.width - blockSize.width),
//                                       (blockSize.width - blockOverlap.width));
//    if (w.rem != 0) {
//        std::stringstream error;
//        error << "checkHOGParameters(): horizontal block number(" << blockNo.width
//              << ") does not fit; cellnumber=" << cellNo.width
//              << " blockOverlap=" << blockOverlap.width;
//        LOG4CXX_ERROR(logger, error.str());
//        throw(error.str());
//    }

//    blockNo = cv::Size(w.quot + 1, h.quot + 1);
}

void PatchFeature::checkCoHOGParameters(
        const cv::Size &cellSize,
        const int offsetNo,
        int &xShift,
        int &yShift,
        int &xShiftRight
        ) {

    if (verbosity > 0) {
        LOG4CXX_INFO(logger, "called checkCoHOGParameters()");
    }

    if (offsetNo < 1) {
        std::stringstream error;
        error << "checkCoHOGParameters(): offsetNo has to be a positive number";
        LOG4CXX_ERROR(logger, error.str());
        throw(error.str());
    }

    /** we have to skip some rows and cols depending on the offset
    offset would be out of bounds otherwise

    0 is the current regarded pixel, everything else shows the
    order of the offset

    numbers with a + indicate that an additional row/col has to be skipped
    ________________________
    |___|___|___|___|___|___|
    |11+|_10|_9_|_8_|_7+|___
    |_12|_4+|_3_|_2+|_6_|__
    |___|___|_0_|_1_|_5_|_
    |___|___|___|___|___|

    yShift, xShift, xShiftRight = count how many rows/cols have to be skipped above/left/right
    count = represents the linear increasing space between two subsequent plus signs
    */
    xShift = 0;
    yShift = 0;
    int counter = 1;
    int i = offsetNo;
    while (i > 1) {
        counter++;
        if ((counter % 2) == 0) {
            yShift++;
        } else {
            xShift++;
        }
        i -= counter;
    }
    xShiftRight = std::max(yShift, 1);

    /** if xShift was incremented as last
        there could be some remainder on the right
        e.g. number 5 in the graphic above */
    if ((counter % 2) == 1
            && (i + counter - (yShift + 1)) > 0) {

        xShiftRight++;
    }

    if (xShift + xShiftRight + 1 > cellSize.width) {
        std::stringstream error;
        error << "checkCoHOGParameters(): offsetNo is too large for the given cellSize";
        LOG4CXX_ERROR(logger, error.str());
        throw(error.str());
    }
}

void PatchFeature::HOGGradients(
        const cv::Mat &im,
        cv::Mat &hGrad,
        const bool useSign,
        const int binNo
        ) {

    if (verbosity > 0) {
        LOG4CXX_INFO(logger, "called HOGGradients()");
    }

    const int binSize = ((useSign ? 360 : 180) / binNo);
    const int w = im.cols;
    const int h = im.rows;
    cv::MatIterator_<BINMAG_VEC_TYPE> binMag = hGrad.begin<BINMAG_VEC_TYPE>();
    cv::MatConstIterator_<SRC_TYPE> p1 = im.begin<SRC_TYPE>() + 1;
    cv::MatConstIterator_<SRC_TYPE> p2 = im.begin<SRC_TYPE>() + w;
    cv::MatConstIterator_<SRC_TYPE> p3 = im.begin<SRC_TYPE>() + 2 * w + 1;
    short gradients[2];

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            gradients[0] = (*p3) - (*p1);
            gradients[1] = (*p2 + 2) - (*p2);

            /** angle ranges from [0, 179] or [0, 359]
                binMag: channel 0 = exact position in bin; channel 1 = magnitude
                atan returns radians; degree = radians * 180 / Pi */
            int angle = (std::atan2((BINMAG_TYPE)gradients[0], (BINMAG_TYPE)gradients[1]) * 180 / M_PI);
            if (angle == 180) {
                angle -= 1;
            }
            if (useSign || angle < 0) {
                angle += 180;
            }

            /** find bin */
            (*binMag)[0] = (angle / (double)binSize);
            (*binMag)[1] = std::sqrt(std::pow(gradients[0],2) + std::pow(gradients[1],2));

            if (verbosity > 4) {
                std::cout << "bin " << (*binMag)[0] << " -- angle " << angle << std::endl;
            }

            p1++;
            p2++;
            p3++;
            binMag++;
        }

        p1 += 2;
        p2 += 2;
        p3 += 2;
    }
}

/**
 * @brief PatchFeature::HOGBinning Creates the histogram of oriented gradients
 *      for a single cell.
 * @param cellGrad
 * @param bins
 * @param useSign
 */
void PatchFeature::HOGBinning(
        const cv::Mat& hGrad,
        MatrixOfMats_t &cellBins,
        const cv::Size cellSize,
        const cv::Size cellNo,
        const int binNo
        ) {

    if (verbosity > 0) {
        LOG4CXX_INFO(logger, "called HOGBinning()");
    }

    cv::Mat cellGrad(hGrad(cv::Rect(0, 0, cellSize.width, cellSize.height)));
    const int w = cellSize.width;
    const int fullW = w * cellNo.width;
    const cv::Size binNoSize(1, binNo);

    cellBins.reserve(cellNo.height);
    cv::Mat bins(binNoSize, BIN_CV_TYPE);
    cv::MatConstIterator_<BINMAG_VEC_TYPE> binMag;

    for (int yy = 0; yy < cellNo.height; yy++) {
        std::vector<cv::Mat> vec;
        vec.reserve(cellNo.width);

        for (int xx = 0; xx < cellNo.width; xx++) {
            bins.setTo(cv::Scalar(0.0));
            binMag = cellGrad.begin<BINMAG_VEC_TYPE>();

            for (int y = 0; y < cellGrad.rows; y++) {
                for (int x = 0; x < cellGrad.cols; x++) {

                    /** add gradient magnitude to bin and its nearest neighbour
                        binMag[0] represents the position in a bin; the center of a bin is at
                        (i * binSize); the borders of a bin are (binSize / 2) away from the center;
                         the first and the last bin are only (binSize / 2) big and are actually one*/
                    const double littleWeight = ((*binMag)[0] - std::floor((*binMag)[0]));
                    bins.at<BIN_TYPE>((*binMag)[0], 0) += (1 - littleWeight) * (*binMag)[1];
                    int i = (*binMag)[0];
                    if (i == bins.rows - 1) {
                        i = 0;
                    } else {
                        i++;
                    }

                    bins.at<BIN_TYPE>(i, 0) += littleWeight * (*binMag)[1];

                    ++binMag;
                }
            }

            vec.push_back(bins.clone());
            if (!(xx + 1 == cellNo.width)) {
                cellGrad.adjustROI(0, 0, -w, w);
            }
        }

        cellBins.push_back(vec);
        if (!(yy + 1 == cellNo.height)) {
            cellGrad.adjustROI(-cellSize.height, cellSize.height, fullW - w, -(fullW - w));
        }
    }
}

void PatchFeature::CoHOGBinning(
        const cv::Mat& hGrad,
        MatrixOfMats_t &cellBins,
        const cv::Size cellSize,
        const cv::Size cellNo,
        const int binNo,
        const int offsetNo,
        const int xShift,
        const int yShift,
        const int xShiftRight
        ) {

    if (verbosity > 0) {
        LOG4CXX_INFO(logger, "called CohogBinning()");
    }

    cv::Mat cellGrad(hGrad(cv::Rect(0, 0, cellSize.width, cellSize.height)));
    const int w = cellSize.width;
    const int fullW = w * cellNo.width;
    const cv::Size binNoSize(binNo, binNo);

    if (verbosity > 4) {
        std::cout << "hGrad=" << hGrad.size() << ", cellNo=" << cellNo
                  << ", xShift=" << xShift << ", yShift=" << yShift
                  << ", xShiftRight=" << xShiftRight
                  << ", cellGradcv::Size=" << cellGrad.size() << std::endl;
    }

    cellBins.reserve(cellNo.height);
    cv::Mat bins(binNoSize, BIN_CV_TYPE);
    cv::MatConstIterator_<BINMAG_VEC_TYPE> fst;
    cv::MatConstIterator_<BINMAG_VEC_TYPE> snd;

    for (int yy = 0; yy < cellNo.height; yy++) {
        std::vector<cv::Mat> vec;
        vec.reserve(cellNo.width);

        for (int xx = 0; xx < cellNo.width; xx++) {
            bins.setTo(cv::Scalar(0.0));
            cv::Mat binVector;

            int offsetX = 1;
            int offsetY = 0;
            int xShiftHelp = -1;
            int yShiftHelp = 1;
            for (int i = 0; i < offsetNo; ++i) {
                fst = cellGrad.begin<BINMAG_VEC_TYPE>() + yShift * w + xShift;
                snd = fst + offsetX - offsetY * w;
                for (int y = yShift; y < cellSize.height; y++) {
                    for (int x = xShift; x < cellSize.width - xShiftRight; x++) {
                        bins.at<BIN_TYPE>((*fst)[0], (*snd)[0]) += (*fst)[1] + (*snd)[1];
                        ++fst;
                        ++snd;
                    }

                    fst += xShiftRight + xShift;
                    snd += xShiftRight + xShift;
                }
                cv::Mat tmp;
                bins.reshape(0, bins.total()).copyTo(tmp);
//                PatchFeatureUtils::MatPrettyPrint(tmp);
                binVector.push_back(tmp);

                if (offsetY < yShiftHelp) {
                    ++offsetY;
                    continue;
                }
                if (xShiftHelp < offsetX) {
                    --offsetX;
                    continue;
                }
                offsetX = offsetY + 1;
                offsetY = 0;
                --xShiftHelp;
                ++yShiftHelp;

            }

            vec.push_back(binVector);
            if (!(xx + 1 == cellNo.width)) {
                cellGrad.adjustROI(0, 0, -w, w);
            }
        }

        cellBins.push_back(vec);
        if (!(yy + 1 == cellNo.height)) {
            cellGrad.adjustROI(-cellSize.height, cellSize.height, fullW - w, -(fullW - w));
        }
    }
}

void PatchFeature::HOGComputeBlocks(
        MatrixOfMats_t &blocks,
        const MatrixOfMats_t &cellBins,
        const cv::Size blockNo,
        const cv::Size blockSize,
        const cv::Size blockOverlap
        ) {

    if (verbosity > 0) {
        LOG4CXX_INFO(logger,  "called HOGComputeBlocks()");
    }

    blocks.resize(blockNo.height);
    for (int i = 0; i < blockNo.height; i++) {
        blocks[i].resize(blockNo.width);
    }
    if (blockSize.height == 1 && blockSize.width == 1) {
        for (int i = 0; i < blockNo.height; i++) {
             for (int j = 0; j < blockNo.width; j++) {
                 const double blockNorm = cv::norm(cellBins[i][j], cv::NORM_L2, cv::noArray());
                 cv::Mat tmp  = cellBins[i][j] / blockNorm;
                 tmp.copyTo(blocks[i][j]);
             }
         }

        return;
    }

    cv::Size binSize = cellBins[0][0].size();
    int cellIdxY = 0;
    int cellIdxX = 0;
    const int reshapedBinH = binSize.width * binSize.height;
    cv::Mat concated(cv::Size(1, blockSize.height * blockSize.width * reshapedBinH), cellBins[0][0].type());
    cv::Mat roi;
    for (int y = 0; y < blockNo.height; y++) {

        cellIdxX = 0;
        for (int x = 0; x < blockNo.width; x++) {

            roi = concated(cv::Rect(0, 0, 1, reshapedBinH));
            for (int i = 0; i < blockSize.height; i++) {
                for (int j = 0; j < blockSize.width; j++) {

                    cv::Mat tmp = cellBins[i + cellIdxY][j + cellIdxX];
                    tmp.reshape(0, tmp.total()).copyTo(roi);
                    roi.adjustROI(-reshapedBinH, reshapedBinH, 0, 0);

                }

                if (verbosity > 4) {
                    std::cout << "concated: \n";
                    PatchFeatureUtils::MatPrettyPrint(concated, concated.cols / 2);
                }
            }

            /** normalizing blocks */
            const double blockNorm = cv::norm(concated, cv::NORM_L2, cv::noArray());
            cv::Mat tmp(concated / blockNorm);
            tmp.copyTo(blocks[y][x]);

            if (verbosity > 4) {
                std::cout << "one block: \n";
                PatchFeatureUtils::MatPrettyPrint(blocks[y][x], blocks[y][x].cols / 2);
            }
            cellIdxX += (blockSize.width - blockOverlap.width);
        }

        cellIdxY += (blockSize.height - blockOverlap.height);
    }
}
