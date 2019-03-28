#ifndef MODIFIEDPCA_H
#define MODIFIEDPCA_H

#define CV_PCA_DATA_AS_ROW   0
#define CV_PCA_DATA_AS_COL   1
#define CV_GEMM_B_T   2
#define CV_REDUCE_SUM   0
#define CV_REDUCE_AVG   1
#define CV_REDUCE_MAX   2
#define CV_REDUCE_MIN   3

#include <iostream>
#include <string.h>
#include "opencv2/core/core.hpp"

namespace faceAnalysis {

 void gemm(
         cv::Mat &src1,
         cv::Mat &src2,
         double alpha,
         cv::InputArray src3,
         double gamma,
         cv::Mat &dst,
         int flags=0
         );

/*!
    Principal Component Analysis

    The class PCA is used to compute the special basis for a set of vectors.
    The basis will consist of eigenvectors of the covariance cv::Mat rix computed
    from the input set of vectors. After PCA is performed, vectors can be transformed from
    the original high-dimensional space to the subspace formed by a few most
    prominent eigenvectors (called the principal components),
    corresponding to the largest eigenvalues of the covariation cv::Mat rix.
    Thus the dimensionality of the vector and the correlation between the coordinates is reduced.

    The following sample is the function that takes two cv::Mat rices. The first one stores the set
    of vectors (a row per vector) that is used to compute PCA, the second one stores another
    "test" set of vectors (a row per vector) that are first compressed with PCA,
    then reconstructed back and then the reconstruction error norm is computed and printed for each vector.

    \code
    using namespace cv;

    PCA compressPCA(const cv::Mat & pcaset, int maxComponents,
                    const cv::Mat & testset, cv::Mat & compressed)
    {
        PCA pca(pcaset, // pass the data
                cv::Mat (), // we do not have a pre-computed mean vector,
                       // so let the PCA engine to compute it
                CV_PCA_DATA_AS_ROW, // indicate that the vectors
                                    // are stored as cv::Mat rix rows
                                    // (use CV_PCA_DATA_AS_COL if the vectors are
                                    // the cv::Mat rix columns)
                maxComponents // specify, how many principal components to retain
                );
        // if there is no test data, just return the computed basis, ready-to-use
        if( !testset.data )
            return pca;
        CV_Assert( testset.cols == pcaset.cols );

        compressed.create(testset.rows, maxComponents, testset.type());

        cv::Mat  reconstructed;
        for( int i = 0; i < testset.rows; i++ )
        {
            cv::Mat  vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
            // compress the vector, the result will be stored
            // in the i-th row of the output cv::Mat rix
            pca.project(vec, coeffs);
            // and then reconstruct it
            pca.backProject(coeffs, reconstructed);
            // and measure the error
            printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
        }
        return pca;
    }
    \endcode
*/
class PCA
{
public:
    //! default constructor
    PCA();
    //! the constructor that performs PCA
    PCA(cv::Mat &data, cv::InputArray mean, int flags, int maxComponents=0);
    PCA(cv::Mat &data, cv::InputArray mean, int flags, double retainedVariance);
    //! operator that performs PCA. The previously stored data, if any, is released
    PCA& operator()(cv::Mat &data, cv::InputArray mean, int flags, int maxComponents=0);
    PCA& computeVar(cv::Mat &data, cv::InputArray mean, int flags, double retainedVariance);
    //! projects vector from the original space to the principal components subspace
    cv::Mat  project(cv::Mat &vec) const;
    //! projects vector from the original space to the principal components subspace
    void project(cv::Mat &vec, cv::Mat &result) const;
    //! reconstructs the original vector from the projection
    cv::Mat  backProject(cv::Mat &vec) const;
    //! reconstructs the original vector from the projection
    void backProject(cv::InputArray vec, cv::OutputArray  result) const;

    cv::Mat  eigenvectors; //!< eigenvectors of the covariation cv::Mat rix
    cv::Mat  eigenvalues; //!< eigenvalues of the covariation cv::Mat rix
    cv::Mat  mean; //!< mean value subtracted before the projection and added after the back projection
};

void PCACompute(cv::InputArray data,  cv::InputOutputArray mean,
                             cv::OutputArray  eigenvectors, int maxComponents=0);

void PCAComputeVar(cv::InputArray data,  cv::InputOutputArray  mean,
                             cv::OutputArray  eigenvectors, double retainedVariance);

void PCAProject(cv::InputArray data, cv::InputArray mean,
                             cv::InputArray eigenvectors, cv::OutputArray  result);

void PCABackProject(cv::InputArray data, cv::InputArray mean,
                                 cv::InputArray eigenvectors, cv::OutputArray  result);

void calcCovarMatrix(
        cv::Mat &_src,
        cv::OutputArray _covar,
        cv::InputOutputArray _mean,
        int flags,
        int ctype
        );

void mulTransposed(
        cv::Mat &src,
        cv::OutputArray _dst,
        bool ata,
        cv::InputArray _delta,
        double scale,
        int dtype
        );

typedef void (*MulTransposedFunc)(const cv::Mat& src, cv::Mat& dst, const cv::Mat& delta, double scale);

void repeat(cv::Mat &src, int ny, int nx, cv::Mat &dst);
//void reduce(
//        cv::Mat &src,
//        cv::OutputArray _dst,
//        int dim,
//        int op,
//        int dtype
//        );

/*!
    Singular Value Decomposition class

    The class is used to compute Singular Value Decomposition of a floating-cv::Point cv::Mat rix and then
    use it to solve least-square problems, under-determined linear systems, invert cv::Mat rices,
    compute condition numbers etc.

    For a bit faster operation you can pass flags=SVD::MODIFY_A|... to modify the decomposed cv::Mat rix
    when it is not necessarily to preserve it. If you want to compute condition number of a cv::Mat rix
    or absolute value of its determinant - you do not need SVD::u or SVD::vt,
    so you can pass flags=SVD::NO_UV|... . Another flag SVD::FULL_UV indicates that the full-size SVD::u and SVD::vt
    must be computed, which is not necessary most of the time.
*/
class SVD
{
public:
    enum { MODIFY_A=1, NO_UV=2, FULL_UV=4 };
    //! the default constructor
    SVD();
    //! the constructor that performs SVD
    SVD( cv::InputArray src, int flags=0 );
    //! the operator that performs SVD. The previously allocated SVD::u, SVD::w are SVD::vt are released.
    SVD& operator ()( cv::InputArray src, int flags=0 );

    //! decomposes cv::Mat rix and stores the results to user-provided cv::Mat rices
    static void compute( cv::InputArray src, cv::OutputArray  w,
                         cv::OutputArray  u, cv::OutputArray  vt, int flags=0 );
    //! computes singular values of a cv::Mat rix
    static void compute( cv::InputArray src, cv::OutputArray  w, int flags=0 );
    //! performs back substitution
    static void backSubst( cv::InputArray w, cv::InputArray u,
                           cv::InputArray vt, cv::InputArray rhs,
                           cv::OutputArray  dst );

    template<typename _Tp, int m, int n, int nm> static void compute( const cv::Matx<_Tp, m, n>& a,
        cv::Matx<_Tp, nm, 1>& w, cv::Matx<_Tp, m, nm>& u, cv::Matx<_Tp, n, nm>& vt );
    template<typename _Tp, int m, int n, int nm> static void compute( const cv::Matx<_Tp, m, n>& a,
        cv::Matx<_Tp, nm, 1>& w );
    template<typename _Tp, int m, int n, int nm, int nb> static void backSubst( const cv::Matx<_Tp, nm, 1>& w,
        const cv::Matx<_Tp, m, nm>& u, const cv::Matx<_Tp, n, nm>& vt, const cv::Matx<_Tp, m, nb>& rhs, cv::Matx<_Tp, n, nb>& dst );

    //! finds dst = arg min_{|dst|=1} |m*dst|
    static void solveZ( cv::InputArray src, cv::OutputArray  dst );
    //! performs back substitution, so that dst is the solution or pseudo-solution of m*dst = rhs, where m is the decomposed cv::Mat rix
    void backSubst( cv::InputArray rhs, cv::OutputArray  dst ) const;

    cv::Mat  u, w, vt;
};

template <typename T>
int computeCumulativeEnergy(const cv::Mat& eigenvalues, double retainedVariance)
{
    CV_DbgAssert( eigenvalues.type() == cv::DataType<T>::type );

    cv::Mat g(eigenvalues.size(), cv::DataType<T>::type);

    for(int ig = 0; ig < g.rows; ig++)
    {
        g.at<T>(ig, 0) = 0;
        for(int im = 0; im <= ig; im++)
        {
            g.at<T>(ig,0) += eigenvalues.at<T>(im,0);
        }
    }

    int L;

    for(L = 0; L < eigenvalues.rows; L++)
    {
        double energy = g.at<T>(L, 0) / g.at<T>(g.rows - 1, 0);
        if(energy > retainedVariance) {
            break;
        }
    }

    L = std::max(2, L);

    return L;
}

template<typename sT, typename dT> static void
MulTransposedR( const cv::Mat& srcmat, cv::Mat& dstmat, const cv::Mat& deltamat, double scale )
{
    int i, j, k;
    const sT* src = (const sT*)srcmat.data;
    dT* dst = (dT*)dstmat.data;
    const dT* delta = (const dT*)deltamat.data;
    size_t srcstep = srcmat.step/sizeof(src[0]);
    size_t dststep = dstmat.step/sizeof(dst[0]);
    size_t deltastep = deltamat.rows > 1 ? deltamat.step/sizeof(delta[0]) : 0;
    int delta_cols = deltamat.cols;
    cv::Size size = srcmat.size();
    dT* tdst = dst;
    dT* col_buf = 0;
    dT* delta_buf = 0;
    int buf_size = size.height*sizeof(dT);
    cv::AutoBuffer<uchar> buf;

    if( delta && delta_cols < size.width )
    {
        assert( delta_cols == 1 );
        buf_size *= 5;
    }
    buf.allocate(buf_size);
    col_buf = (dT*)(uchar*)buf;

    if( delta && delta_cols < size.width )
    {
        delta_buf = col_buf + size.height;
        for( i = 0; i < size.height; i++ )
            delta_buf[i*4] = delta_buf[i*4+1] =
                delta_buf[i*4+2] = delta_buf[i*4+3] = delta[i*deltastep];
        delta = delta_buf;
        deltastep = deltastep ? 4 : 0;
    }

    if( !delta )
        for( i = 0; i < size.width; i++, tdst += dststep )
        {
            for( k = 0; k < size.height; k++ )
                col_buf[k] = src[k*srcstep+i];

            for( j = i; j <= size.width - 4; j += 4 )
            {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                const sT *tsrc = src + j;

                for( k = 0; k < size.height; k++, tsrc += srcstep )
                {
                    double a = col_buf[k];
                    s0 += a * tsrc[0];
                    s1 += a * tsrc[1];
                    s2 += a * tsrc[2];
                    s3 += a * tsrc[3];
                }

                tdst[j] = (dT)(s0*scale);
                tdst[j+1] = (dT)(s1*scale);
                tdst[j+2] = (dT)(s2*scale);
                tdst[j+3] = (dT)(s3*scale);
            }

            for( ; j < size.width; j++ )
            {
                double s0 = 0;
                const sT *tsrc = src + j;

                for( k = 0; k < size.height; k++, tsrc += srcstep )
                    s0 += (double)col_buf[k] * tsrc[0];

                tdst[j] = (dT)(s0*scale);
            }
        }
    else
        for( i = 0; i < size.width; i++, tdst += dststep )
        {
            if( !delta_buf )
                for( k = 0; k < size.height; k++ )
                    col_buf[k] = src[k*srcstep+i] - delta[k*deltastep+i];
            else
                for( k = 0; k < size.height; k++ )
                    col_buf[k] = src[k*srcstep+i] - delta_buf[k*deltastep];

            for( j = i; j <= size.width - 4; j += 4 )
            {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                const sT *tsrc = src + j;
                const dT *d = delta_buf ? delta_buf : delta + j;

                for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                {
                    double a = col_buf[k];
                    s0 += a * (tsrc[0] - d[0]);
                    s1 += a * (tsrc[1] - d[1]);
                    s2 += a * (tsrc[2] - d[2]);
                    s3 += a * (tsrc[3] - d[3]);
                }

                tdst[j] = (dT)(s0*scale);
                tdst[j+1] = (dT)(s1*scale);
                tdst[j+2] = (dT)(s2*scale);
                tdst[j+3] = (dT)(s3*scale);
            }

            for( ; j < size.width; j++ )
            {
                double s0 = 0;
                const sT *tsrc = src + j;
                const dT *d = delta_buf ? delta_buf : delta + j;

                for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                    s0 += (double)col_buf[k] * (tsrc[0] - d[0]);

                tdst[j] = (dT)(s0*scale);
            }
        }
}


template<typename sT, typename dT> static void
MulTransposedL( const cv::Mat& srcmat, cv::Mat& dstmat, const cv::Mat& deltamat, double scale )
{
    int i, j, k;
    const sT* src = (const sT*)srcmat.data;
    dT* dst = (dT*)dstmat.data;
    const dT* delta = (const dT*)deltamat.data;
    size_t srcstep = srcmat.step/sizeof(src[0]);
    size_t dststep = dstmat.step/sizeof(dst[0]);
    size_t deltastep = deltamat.rows > 1 ? deltamat.step/sizeof(delta[0]) : 0;
    int delta_cols = deltamat.cols;
    cv::Size size = srcmat.size();
    dT* tdst = dst;

    if( !delta )
        for( i = 0; i < size.height; i++, tdst += dststep )
            for( j = i; j < size.height; j++ )
            {
                double s = 0;
                const sT *tsrc1 = src + i*srcstep;
                const sT *tsrc2 = src + j*srcstep;

                for( k = 0; k <= size.width - 4; k += 4 )
                    s += (double)tsrc1[k]*tsrc2[k] + (double)tsrc1[k+1]*tsrc2[k+1] +
                         (double)tsrc1[k+2]*tsrc2[k+2] + (double)tsrc1[k+3]*tsrc2[k+3];
                for( ; k < size.width; k++ )
                    s += (double)tsrc1[k] * tsrc2[k];
                tdst[j] = (dT)(s*scale);
            }
    else
    {
        dT delta_buf[4];
        int delta_shift = delta_cols == size.width ? 4 : 0;
        cv::AutoBuffer<uchar> buf(size.width*sizeof(dT));
        dT* row_buf = (dT*)(uchar*)buf;

        for( i = 0; i < size.height; i++, tdst += dststep )
        {
            const sT *tsrc1 = src + i*srcstep;
            const dT *tdelta1 = delta + i*deltastep;

            if( delta_cols < size.width )
                for( k = 0; k < size.width; k++ )
                    row_buf[k] = tsrc1[k] - tdelta1[0];
            else
                for( k = 0; k < size.width; k++ )
                    row_buf[k] = tsrc1[k] - tdelta1[k];

            for( j = i; j < size.height; j++ )
            {
                double s = 0;
                const sT *tsrc2 = src + j*srcstep;
                const dT *tdelta2 = delta + j*deltastep;
                if( delta_cols < size.width )
                {
                    delta_buf[0] = delta_buf[1] =
                        delta_buf[2] = delta_buf[3] = tdelta2[0];
                    tdelta2 = delta_buf;
                }
                for( k = 0; k <= size.width-4; k += 4, tdelta2 += delta_shift )
                    s += (double)row_buf[k]*(tsrc2[k] - tdelta2[0]) +
                         (double)row_buf[k+1]*(tsrc2[k+1] - tdelta2[1]) +
                         (double)row_buf[k+2]*(tsrc2[k+2] - tdelta2[2]) +
                         (double)row_buf[k+3]*(tsrc2[k+3] - tdelta2[3]);
                for( ; k < size.width; k++, tdelta2++ )
                    s += (double)row_buf[k]*(tsrc2[k] - tdelta2[0]);
                tdst[j] = (dT)(s*scale);
            }
        }
    }
}

}

#endif /*MODIFIEDPCA_H*/
