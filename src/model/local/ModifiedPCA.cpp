#include "model/local/ModifiedPCA.h"

static void
GEMM_CopyBlock( const uchar* src, size_t src_step,
                uchar* dst, size_t dst_step,
                cv::Size size, size_t pix_size )
{
    int j;
    size.width *= (int)(pix_size / sizeof(int));

    for( ; size.height--; src += src_step, dst += dst_step )
    {
        j=0;
         #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            int t0 = ((const int*)src)[j];
            int t1 = ((const int*)src)[j+1];
            ((int*)dst)[j] = t0;
            ((int*)dst)[j+1] = t1;
            t0 = ((const int*)src)[j+2];
            t1 = ((const int*)src)[j+3];
            ((int*)dst)[j+2] = t0;
            ((int*)dst)[j+3] = t1;
        }
        #endif
        for( ; j < size.width; j++ )
            ((int*)dst)[j] = ((const int*)src)[j];
    }
}


static void
GEMM_TransposeBlock( const uchar* src, size_t src_step,
                     uchar* dst, size_t dst_step,
                     cv::Size size, size_t pix_size )
{
    int i, j;
    for( i = 0; i < size.width; i++, dst += dst_step, src += pix_size )
    {
        const uchar* _src = src;
        switch( pix_size )
        {
        case sizeof(int):
            for( j = 0; j < size.height; j++, _src += src_step )
                ((int*)dst)[j] = ((int*)_src)[0];
            break;
        case sizeof(int)*2:
            for( j = 0; j < size.height*2; j += 2, _src += src_step )
            {
                int t0 = ((int*)_src)[0];
                int t1 = ((int*)_src)[1];
                ((int*)dst)[j] = t0;
                ((int*)dst)[j+1] = t1;
            }
            break;
        case sizeof(int)*4:
            for( j = 0; j < size.height*4; j += 4, _src += src_step )
            {
                int t0 = ((int*)_src)[0];
                int t1 = ((int*)_src)[1];
                ((int*)dst)[j] = t0;
                ((int*)dst)[j+1] = t1;
                t0 = ((int*)_src)[2];
                t1 = ((int*)_src)[3];
                ((int*)dst)[j+2] = t0;
                ((int*)dst)[j+3] = t1;
            }
            break;
        default:
            assert(0);
            return;
        }
    }
}


template<typename T, typename WT> static void
GEMMSingleMul( const T* a_data, size_t a_step,
               const T* b_data, size_t b_step,
               const T* c_data, size_t c_step,
               T* d_data, size_t d_step,
               cv::Size a_size, cv::Size d_size,
               double alpha, double beta, int flags )
{
    int i, j, k, n = a_size.width, m = d_size.width, drows = d_size.height;
    const T *_a_data = a_data, *_b_data = b_data, *_c_data = c_data;
    cv::AutoBuffer<T> _a_buf;
    T* a_buf = 0;
    size_t a_step0, a_step1, c_step0, c_step1, t_step;

    a_step /= sizeof(a_data[0]);
    b_step /= sizeof(b_data[0]);
    c_step /= sizeof(c_data[0]);
    d_step /= sizeof(d_data[0]);
    a_step0 = a_step;
    a_step1 = 1;

    if( !c_data )
        c_step0 = c_step1 = 0;
    else if( !(flags & cv::GEMM_3_T) )
        c_step0 = c_step, c_step1 = 1;
    else
        c_step0 = 1, c_step1 = c_step;

    if( flags & cv::GEMM_1_T )
    {
        CV_SWAP( a_step0, a_step1, t_step );
        n = a_size.height;
        if( a_step > 1 && n > 1 )
        {
            _a_buf.allocate(n);
            a_buf = _a_buf;
        }
    }

    if( n == 1 ) /* external product */
    {
        cv::AutoBuffer<T> _b_buf;
        T* b_buf = 0;

        if( a_step > 1 && a_size.height > 1 )
        {
            _a_buf.allocate(drows);
            a_buf = _a_buf;
            for( k = 0; k < drows; k++ )
                a_buf[k] = a_data[a_step*k];
            a_data = a_buf;
        }

        if( b_step > 1 )
        {
            _b_buf.allocate(d_size.width);
            b_buf = _b_buf;
            for( j = 0; j < d_size.width; j++ )
                b_buf[j] = b_data[j*b_step];
            b_data = b_buf;
        }

        for( i = 0; i < drows; i++, _c_data += c_step0, d_data += d_step )
        {
            WT al = WT(a_data[i])*alpha;
            c_data = _c_data;
            for( j = 0; j <= d_size.width - 2; j += 2, c_data += 2*c_step1 )
            {
                WT s0 = al*WT(b_data[j]);
                WT s1 = al*WT(b_data[j+1]);
                if( !c_data )
                {
                    d_data[j] = T(s0);
                    d_data[j+1] = T(s1);
                }
                else
                {
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
                    d_data[j+1] = T(s1 + WT(c_data[c_step1])*beta);
                }
            }

            for( ; j < d_size.width; j++, c_data += c_step1 )
            {
                WT s0 = al*WT(b_data[j]);
                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else if( flags & cv::GEMM_2_T ) /* A * Bt */
    {
        for( i = 0; i < drows; i++, _a_data += a_step0, _c_data += c_step0, d_data += d_step )
        {
            a_data = _a_data;
            b_data = _b_data;
            c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < d_size.width; j++, b_data += b_step,
                                               c_data += c_step1 )
            {
                WT s0(0), s1(0), s2(0), s3(0);
                k = 0;
                 #if CV_ENABLE_UNROLLED
                for( ; k <= n - 4; k += 4 )
                {
                    s0 += WT(a_data[k])*WT(b_data[k]);
                    s1 += WT(a_data[k+1])*WT(b_data[k+1]);
                    s2 += WT(a_data[k+2])*WT(b_data[k+2]);
                    s3 += WT(a_data[k+3])*WT(b_data[k+3]);
                }
                #endif
                for( ; k < n; k++ )
                    s0 += WT(a_data[k])*WT(b_data[k]);
                s0 = (s0+s1+s2+s3)*alpha;

                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else if( d_size.width*sizeof(d_data[0]) <= 1600 )
    {
        for( i = 0; i < drows; i++, _a_data += a_step0,
                                    _c_data += c_step0,
                                    d_data += d_step )
        {
            a_data = _a_data, c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j <= m - 4; j += 4, c_data += 4*c_step1 )
            {
                const T* b = _b_data + j;
                WT s0(0), s1(0), s2(0), s3(0);

                for( k = 0; k < n; k++, b += b_step )
                {
                    WT a(a_data[k]);
                    s0 += a * WT(b[0]); s1 += a * WT(b[1]);
                    s2 += a * WT(b[2]); s3 += a * WT(b[3]);
                }

                if( !c_data )
                {
                    d_data[j] = T(s0*alpha);
                    d_data[j+1] = T(s1*alpha);
                    d_data[j+2] = T(s2*alpha);
                    d_data[j+3] = T(s3*alpha);
                }
                else
                {
                    s0 = s0*alpha; s1 = s1*alpha;
                    s2 = s2*alpha; s3 = s3*alpha;
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
                    d_data[j+1] = T(s1 + WT(c_data[c_step1])*beta);
                    d_data[j+2] = T(s2 + WT(c_data[c_step1*2])*beta);
                    d_data[j+3] = T(s3 + WT(c_data[c_step1*3])*beta);
                }
            }

            for( ; j < m; j++, c_data += c_step1 )
            {
                const T* b = _b_data + j;
                WT s0(0);

                for( k = 0; k < n; k++, b += b_step )
                    s0 += WT(a_data[k]) * WT(b[0]);

                s0 = s0*alpha;
                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else
    {
        cv::AutoBuffer<WT> _d_buf(m);
        WT* d_buf = _d_buf;

        for( i = 0; i < drows; i++, _a_data += a_step0, _c_data += c_step0, d_data += d_step )
        {
            a_data = _a_data;
            b_data = _b_data;
            c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = _a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < m; j++ )
                d_buf[j] = WT(0);

            for( k = 0; k < n; k++, b_data += b_step )
            {
                WT al(a_data[k]);
                j=0;
                 #if CV_ENABLE_UNROLLED
                for(; j <= m - 4; j += 4 )
                {
                    WT t0 = d_buf[j] + WT(b_data[j])*al;
                    WT t1 = d_buf[j+1] + WT(b_data[j+1])*al;
                    d_buf[j] = t0;
                    d_buf[j+1] = t1;
                    t0 = d_buf[j+2] + WT(b_data[j+2])*al;
                    t1 = d_buf[j+3] + WT(b_data[j+3])*al;
                    d_buf[j+2] = t0;
                    d_buf[j+3] = t1;
                }
                #endif
                for( ; j < m; j++ )
                    d_buf[j] += WT(b_data[j])*al;
            }

            if( !c_data )
                for( j = 0; j < m; j++ )
                    d_data[j] = T(d_buf[j]*alpha);
            else
                for( j = 0; j < m; j++, c_data += c_step1 )
                {
                    WT t = d_buf[j]*alpha;
                    d_data[j] = T(t + WT(c_data[0])*beta);
                }
        }
    }
}


template<typename T, typename WT> static void
GEMMBlockMul( const T* a_data, size_t a_step,
              const T* b_data, size_t b_step,
              WT* d_data, size_t d_step,
              cv::Size a_size, cv::Size d_size, int flags )
{
    int i, j, k, n = a_size.width, m = d_size.width;
    const T *_a_data = a_data, *_b_data = b_data;
    cv::AutoBuffer<T> _a_buf;
    T* a_buf = 0;
    size_t a_step0, a_step1, t_step;
    int do_acc = flags & 16;

    a_step /= sizeof(a_data[0]);
    b_step /= sizeof(b_data[0]);
    d_step /= sizeof(d_data[0]);

    a_step0 = a_step;
    a_step1 = 1;

    if( flags & cv::GEMM_1_T )
    {
        CV_SWAP( a_step0, a_step1, t_step );
        n = a_size.height;
        _a_buf.allocate(n);
        a_buf = _a_buf;
    }

    if( flags & cv::GEMM_2_T )
    {
        /* second operand is transposed */
        for( i = 0; i < d_size.height; i++, _a_data += a_step0, d_data += d_step )
        {
            a_data = _a_data; b_data = _b_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < d_size.width; j++, b_data += b_step )
            {
                WT s0 = do_acc ? d_data[j]:WT(0), s1(0);
                for( k = 0; k <= n - 2; k += 2 )
                {
                    s0 += WT(a_data[k])*WT(b_data[k]);
                    s1 += WT(a_data[k+1])*WT(b_data[k+1]);
                }

                for( ; k < n; k++ )
                    s0 += WT(a_data[k])*WT(b_data[k]);

                d_data[j] = s0 + s1;
            }
        }
    }
    else
    {
        for( i = 0; i < d_size.height; i++, _a_data += a_step0, d_data += d_step )
        {
            a_data = _a_data, b_data = _b_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j <= m - 4; j += 4 )
            {
                WT s0, s1, s2, s3;
                const T* b = b_data + j;

                if( do_acc )
                {
                    s0 = d_data[j]; s1 = d_data[j+1];
                    s2 = d_data[j+2]; s3 = d_data[j+3];
                }
                else
                    s0 = s1 = s2 = s3 = WT(0);

                for( k = 0; k < n; k++, b += b_step )
                {
                    WT a(a_data[k]);
                    s0 += a * WT(b[0]); s1 += a * WT(b[1]);
                    s2 += a * WT(b[2]); s3 += a * WT(b[3]);
                }

                d_data[j] = s0; d_data[j+1] = s1;
                d_data[j+2] = s2; d_data[j+3] = s3;
            }

            for( ; j < m; j++ )
            {
                const T* b = b_data + j;
                WT s0 = do_acc ? d_data[j] : WT(0);

                for( k = 0; k < n; k++, b += b_step )
                    s0 += WT(a_data[k]) * WT(b[0]);

                d_data[j] = s0;
            }
        }
    }
}


template<typename T, typename WT> static void
GEMMStore( const T* c_data, size_t c_step,
           const WT* d_buf, size_t d_buf_step,
           T* d_data, size_t d_step, cv::Size d_size,
           double alpha, double beta, int flags )
{
    const T* _c_data = c_data;
    int j;
    size_t c_step0, c_step1;

    c_step /= sizeof(c_data[0]);
    d_buf_step /= sizeof(d_buf[0]);
    d_step /= sizeof(d_data[0]);

    if( !c_data )
        c_step0 = c_step1 = 0;
    else if( !(flags & cv::GEMM_3_T) )
        c_step0 = c_step, c_step1 = 1;
    else
        c_step0 = 1, c_step1 = c_step;

    for( ; d_size.height--; _c_data += c_step0, d_buf += d_buf_step, d_data += d_step )
    {
        if( _c_data )
        {
            c_data = _c_data;
            j=0;
             #if CV_ENABLE_UNROLLED
            for(; j <= d_size.width - 4; j += 4, c_data += 4*c_step1 )
            {
                WT t0 = alpha*d_buf[j];
                WT t1 = alpha*d_buf[j+1];
                t0 += beta*WT(c_data[0]);
                t1 += beta*WT(c_data[c_step1]);
                d_data[j] = T(t0);
                d_data[j+1] = T(t1);
                t0 = alpha*d_buf[j+2];
                t1 = alpha*d_buf[j+3];
                t0 += beta*WT(c_data[c_step1*2]);
                t1 += beta*WT(c_data[c_step1*3]);
                d_data[j+2] = T(t0);
                d_data[j+3] = T(t1);
            }
            #endif
            for( ; j < d_size.width; j++, c_data += c_step1 )
            {
                WT t0 = alpha*d_buf[j];
                d_data[j] = T(t0 + WT(c_data[0])*beta);
            }
        }
        else
        {
            j = 0;
             #if CV_ENABLE_UNROLLED
            for( ; j <= d_size.width - 4; j += 4 )
            {
                WT t0 = alpha*d_buf[j];
                WT t1 = alpha*d_buf[j+1];
                d_data[j] = T(t0);
                d_data[j+1] = T(t1);
                t0 = alpha*d_buf[j+2];
                t1 = alpha*d_buf[j+3];
                d_data[j+2] = T(t0);
                d_data[j+3] = T(t1);
            }
            #endif
            for( ; j < d_size.width; j++ )
                d_data[j] = T(alpha*d_buf[j]);
        }
    }
}



typedef void (*GEMMSingleMulFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, const void* src3, size_t step3,
                   void* dst, size_t dststep, cv::Size srcsize, cv::Size dstsize,
                   double alpha, double beta, int flags );


typedef void (*GEMMBlockMulFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, void* dst, size_t dststep,
                   cv::Size srcsize, cv::Size dstsize, int flags );

typedef void (*GEMMStoreFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, void* dst, size_t dststep,
                   cv::Size dstsize, double alpha, double beta, int flags );

static void GEMMSingleMul_32f( const float* a_data, size_t a_step,
              const float* b_data, size_t b_step,
              const float* c_data, size_t c_step,
              float* d_data, size_t d_step,
              cv::Size a_size, cv::Size d_size,
              double alpha, double beta, int flags )
{
    GEMMSingleMul<float,double>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}

static void GEMMSingleMul_64f( const double* a_data, size_t a_step,
                              const double* b_data, size_t b_step,
                              const double* c_data, size_t c_step,
                              double* d_data, size_t d_step,
                              cv::Size a_size, cv::Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<double,double>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}


static void GEMMSingleMul_32fc( const cv::Complexf* a_data, size_t a_step,
                              const cv::Complexf* b_data, size_t b_step,
                              const cv::Complexf* c_data, size_t c_step,
                              cv::Complexf* d_data, size_t d_step,
                              cv::Size a_size, cv::Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<cv::Complexf,cv::Complexd>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}

static void GEMMSingleMul_64fc( const cv::Complexd* a_data, size_t a_step,
                              const cv::Complexd* b_data, size_t b_step,
                              const cv::Complexd* c_data, size_t c_step,
                              cv::Complexd* d_data, size_t d_step,
                              cv::Size a_size, cv::Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<cv::Complexd,cv::Complexd>(a_data, a_step, b_data, b_step, c_data,
                                 c_step, d_data, d_step, a_size, d_size,
                                 alpha, beta, flags);
}

static void GEMMBlockMul_32f( const float* a_data, size_t a_step,
             const float* b_data, size_t b_step,
             double* d_data, size_t d_step,
             cv::Size a_size, cv::Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_64f( const double* a_data, size_t a_step,
                             const double* b_data, size_t b_step,
                             double* d_data, size_t d_step,
                             cv::Size a_size, cv::Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_32fc( const cv::Complexf* a_data, size_t a_step,
                             const cv::Complexf* b_data, size_t b_step,
                             cv::Complexd* d_data, size_t d_step,
                             cv::Size a_size, cv::Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_64fc( const cv::Complexd* a_data, size_t a_step,
                             const cv::Complexd* b_data, size_t b_step,
                             cv::Complexd* d_data, size_t d_step,
                             cv::Size a_size, cv::Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMStore_32f( const float* c_data, size_t c_step,
          const double* d_buf, size_t d_buf_step,
          float* d_data, size_t d_step, cv::Size d_size,
          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_64f( const double* c_data, size_t c_step,
                      const double* d_buf, size_t d_buf_step,
                      double* d_data, size_t d_step, cv::Size d_size,
                      double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_32fc( const cv::Complexf* c_data, size_t c_step,
                          const cv::Complexd* d_buf, size_t d_buf_step,
                          cv::Complexf* d_data, size_t d_step, cv::Size d_size,
                          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_64fc( const cv::Complexd* c_data, size_t c_step,
                          const cv::Complexd* d_buf, size_t d_buf_step,
                          cv::Complexd* d_data, size_t d_step, cv::Size d_size,
                          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}

void faceAnalysis::gemm(
        cv::Mat &A,
        cv::Mat &B,
        double alpha,
        cv::InputArray MatC,
        double beta,
        cv::Mat &dst,
        int flags
        ) {
std::cout << "tick gemm" << std::endl;
    const int block_lin_size = 128;
    const int block_size = block_lin_size * block_lin_size;

    static double zero[] = {0,0,0,0};
    static float zerof[] = {0,0,0,0};

    cv::Mat C = beta != 0 ? MatC.getMat() : cv::Mat();
    cv::Size a_size = A.size(), d_size;
    int i, len = 0, type = A.type();

    CV_Assert( type == B.type() && (type == CV_32FC1 || type == CV_64FC1 || type == CV_32FC2 || type == CV_64FC2) );

    switch( flags & (cv::GEMM_1_T|cv::GEMM_2_T) )
    {
    case 0:
        d_size = cv::Size( B.cols, a_size.height );
        len = B.rows;
        CV_Assert( a_size.width == len );
        break;
    case 1:
        d_size = cv::Size( B.cols, a_size.width );
        len = B.rows;
        CV_Assert( a_size.height == len );
        break;
    case 2:
        d_size = cv::Size( B.rows, a_size.height );
        len = B.cols;
        CV_Assert( a_size.width == len );
        break;
    case 3:
        d_size = cv::Size( B.rows, a_size.width );
        len = B.cols;
        CV_Assert( a_size.height == len );
        break;
    }

    if( C.data )
    {
        CV_Assert( C.type() == type &&
            (((flags& cv::GEMM_3_T) == 0 && C.rows == d_size.height && C.cols == d_size.width) ||
             ((flags&cv::GEMM_3_T) != 0 && C.rows == d_size.width && C.cols == d_size.height)));
    }

//    _MatD.create( d_size.height, d_size.width, type );
//    cv::Mat dst = _MatD.getMat();
    if( (flags & cv::GEMM_3_T) != 0 && C.data == dst.data )
    {
        transpose( C, C );
        flags &= ~cv::GEMM_3_T;
    }

    if( flags == 0 && 2 <= len && len <= 4 && (len == d_size.width || len == d_size.height) )
    {
        if( type == CV_32F )
        {
            float* d = (float*)dst.data;
            const float *a = (const float*)A.data,
                        *b = (const float*)B.data,
                        *c = (const float*)C.data;
            size_t d_step = dst.step/sizeof(d[0]),
                a_step = A.step/sizeof(a[0]),
                b_step = B.step/sizeof(b[0]),
                c_step = C.data ? C.step/sizeof(c[0]) : 0;

            if( !c )
                c = zerof;

            switch( len )
            {
            case 2:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                    }
                }
                else
                    break;
                return;
            case 3:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1];
                        float t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                        d[2] = (float)(t2*alpha + c[2]*beta);
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] + a[a_step+2]*b[b_step*2];
                        float t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] + a[a_step*2+2]*b[b_step*2];

                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                        d[d_step*2] = (float)(t2*alpha + c[c_step*2]*beta);
                    }
                }
                else
                    break;
                return;
            case 4:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1] + a[3]*b[b_step*3+1];
                        float t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2] + a[3]*b[b_step*3+2];
                        float t3 = a[0]*b[3] + a[1]*b[b_step+3] + a[2]*b[b_step*2+3] + a[3]*b[b_step*3+3];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                        d[2] = (float)(t2*alpha + c[2]*beta);
                        d[3] = (float)(t3*alpha + c[3]*beta);
                    }
                }
                else if( len <= 16 && a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] +
                                   a[a_step+2]*b[b_step*2] + a[a_step+3]*b[b_step*3];
                        float t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] +
                                   a[a_step*2+2]*b[b_step*2] + a[a_step*2+3]*b[b_step*3];
                        float t3 = a[a_step*3]*b[0] + a[a_step*3+1]*b[b_step] +
                                   a[a_step*3+2]*b[b_step*2] + a[a_step*3+3]*b[b_step*3];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                        d[d_step*2] = (float)(t2*alpha + c[c_step*2]*beta);
                        d[d_step*3] = (float)(t3*alpha + c[c_step*3]*beta);
                    }
                }
                else
                    break;
                return;
            }
        }

        if( type == CV_64F )
        {
            double* d = (double*)dst.data;
            const double *a = (const double*)A.data,
                         *b = (const double*)B.data,
                         *c = (const double*)C.data;
            size_t d_step = dst.step/sizeof(d[0]),
                a_step = A.step/sizeof(a[0]),
                b_step = B.step/sizeof(b[0]),
                c_step = C.data ? C.step/sizeof(c[0]) : 0;
            if( !c )
                c = zero;

            switch( len )
            {
            case 2:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step];
                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                    }
                }
                else
                    break;
                return;
            case 3:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1];
                        double t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                        d[2] = t2*alpha + c[2]*beta;
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] + a[a_step+2]*b[b_step*2];
                        double t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] + a[a_step*2+2]*b[b_step*2];

                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                        d[d_step*2] = t2*alpha + c[c_step*2]*beta;
                    }
                }
                else
                    break;
                return;
            case 4:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1] + a[3]*b[b_step*3+1];
                        double t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2] + a[3]*b[b_step*3+2];
                        double t3 = a[0]*b[3] + a[1]*b[b_step+3] + a[2]*b[b_step*2+3] + a[3]*b[b_step*3+3];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                        d[2] = t2*alpha + c[2]*beta;
                        d[3] = t3*alpha + c[3]*beta;
                    }
                }
                else if( d_size.width <= 16 && a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] +
                                    a[a_step+2]*b[b_step*2] + a[a_step+3]*b[b_step*3];
                        double t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] +
                                    a[a_step*2+2]*b[b_step*2] + a[a_step*2+3]*b[b_step*3];
                        double t3 = a[a_step*3]*b[0] + a[a_step*3+1]*b[b_step] +
                                    a[a_step*3+2]*b[b_step*2] + a[a_step*3+3]*b[b_step*3];
                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                        d[d_step*2] = t2*alpha + c[c_step*2]*beta;
                        d[d_step*3] = t3*alpha + c[c_step*3]*beta;
                    }
                }
                else
                    break;
                return;
            }
        }
    }

    {
    size_t b_step = B.step;
    GEMMSingleMulFunc singleMulFunc;
    GEMMBlockMulFunc blockMulFunc;
    GEMMStoreFunc storeFunc;
    cv::Mat tMat;
    const uchar* Cdata = C.data;
    size_t Cstep = C.data ? (size_t)C.step : 0;
    cv::AutoBuffer<uchar> buf;

    if( type == CV_32FC1 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_32f;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_32f;
        storeFunc = (GEMMStoreFunc)GEMMStore_32f;
    }
    else if( type == CV_64FC1 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_64f;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_64f;
        storeFunc = (GEMMStoreFunc)GEMMStore_64f;
    }
    else if( type == CV_32FC2 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_32fc;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_32fc;
        storeFunc = (GEMMStoreFunc)GEMMStore_32fc;
    }
    else
    {
        CV_Assert( type == CV_64FC2 );
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_64fc;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_64fc;
        storeFunc = (GEMMStoreFunc)GEMMStore_64fc;
    }

    if( dst.data == A.data || dst.data == B.data )
    {
        buf.allocate(d_size.width*d_size.height*CV_ELEM_SIZE(type));
        tMat = cv::Mat(d_size.height, d_size.width, type, (uchar*)buf );
        dst = tMat;
    }

    if( (d_size.width == 1 || len == 1) && !(flags & cv::GEMM_2_T) && B.isContinuous() )
    {
        b_step = d_size.width == 1 ? 0 : CV_ELEM_SIZE(type);
        flags |= cv::GEMM_2_T;
    }

    if( ((d_size.height <= block_lin_size/2 || d_size.width <= block_lin_size/2) &&
        len <= 10000) || len <= 10 ||
        (d_size.width <= block_lin_size &&
        d_size.height <= block_lin_size && len <= block_lin_size) )
    {
        singleMulFunc( A.data, A.step, B.data, b_step, Cdata, Cstep,
                       dst.data, dst.step, a_size, d_size, alpha, beta, flags );
    }
    else
    {
        int is_a_t = flags & cv::GEMM_1_T;
        int is_b_t = flags & cv::GEMM_2_T;
        int elem_size = CV_ELEM_SIZE(type);
        int dk0_1, dk0_2;
        int a_buf_size = 0, b_buf_size, d_buf_size;
        uchar* a_buf = 0;
        uchar* b_buf = 0;
        uchar* d_buf = 0;
        int j, k, di = 0, dj = 0, dk = 0;
        int dm0, dn0, dk0;
        size_t a_step0, a_step1, b_step0, b_step1, c_step0, c_step1;
        int work_elem_size = elem_size << 0; //<<<<<<<<<<<<<_____----------------------(cv::CV_Mat_DEPTH(type) == CV_32F ? 1 : 0);

        if( !is_a_t )
            a_step0 = A.step, a_step1 = elem_size;
        else
            a_step0 = elem_size, a_step1 = A.step;

        if( !is_b_t )
            b_step0 = b_step, b_step1 = elem_size;
        else
            b_step0 = elem_size, b_step1 = b_step;

        if( !C.data )
        {
            c_step0 = c_step1 = 0;
            flags &= ~cv::GEMM_3_T;
        }
        else if( !(flags & cv::GEMM_3_T) )
            c_step0 = C.step, c_step1 = elem_size;
        else
            c_step0 = elem_size, c_step1 = C.step;

        dm0 = std::min( block_lin_size, d_size.height );
        dn0 = std::min( block_lin_size, d_size.width );
        dk0_1 = block_size / dm0;
        dk0_2 = block_size / dn0;
        dk0 = std::min( dk0_1, dk0_2 );
        dk0 = std::min( dk0, len );
        if( dk0*dm0 > block_size )
            dm0 = block_size / dk0;
        if( dk0*dn0 > block_size )
            dn0 = block_size / dk0;

        dk0_1 = (dn0+dn0/8+2) & -2;
        b_buf_size = (dk0+dk0/8+1)*dk0_1*elem_size;
        d_buf_size = (dk0+dk0/8+1)*dk0_1*work_elem_size;

        if( is_a_t )
        {
            a_buf_size = (dm0+dm0/8+1)*((dk0+dk0/8+2)&-2)*elem_size;
            flags &= ~cv::GEMM_1_T;
        }

        buf.allocate(a_buf_size + b_buf_size + d_buf_size);
        d_buf = (uchar*)buf;
        b_buf = d_buf + d_buf_size;

        if( is_a_t )
            a_buf = b_buf + b_buf_size;

        for( i = 0; i < d_size.height; i += di )
        {
            di = dm0;
            if( i + di >= d_size.height || 8*(i + di) + di > 8*d_size.height )
                di = d_size.height - i;

            for( j = 0; j < d_size.width; j += dj )
            {
                uchar* _d = dst.data + i*dst.step + j*elem_size;
                const uchar* _c = Cdata + i*c_step0 + j*c_step1;
                size_t _d_step = dst.step;
                dj = dn0;

                if( j + dj >= d_size.width || 8*(j + dj) + dj > 8*d_size.width )
                    dj = d_size.width - j;

                flags &= 15;
                if( dk0 < len )
                {
                    _d = d_buf;
                    _d_step = dj*work_elem_size;
                }

                for( k = 0; k < len; k += dk )
                {
                    const uchar* _a = A.data + i*a_step0 + k*a_step1;
                    size_t _a_step = A.step;
                    const uchar* _b = B.data + k*b_step0 + j*b_step1;
                    size_t _b_step = b_step;
                    cv::Size a_bl_size;

                    dk = dk0;
                    if( k + dk >= len || 8*(k + dk) + dk > 8*len )
                        dk = len - k;

                    if( !is_a_t )
                        a_bl_size.width = dk, a_bl_size.height = di;
                    else
                        a_bl_size.width = di, a_bl_size.height = dk;

                    if( a_buf && is_a_t )
                    {
                        _a_step = dk*elem_size;
                        GEMM_TransposeBlock( _a, A.step, a_buf, _a_step, a_bl_size, elem_size );
                        std::swap( a_bl_size.width, a_bl_size.height );
                        _a = a_buf;
                    }

                    if( dj < d_size.width )
                    {
                        cv::Size b_size;
                        if( !is_b_t )
                            b_size.width = dj, b_size.height = dk;
                        else
                            b_size.width = dk, b_size.height = dj;

                        _b_step = b_size.width*elem_size;
                        GEMM_CopyBlock( _b, b_step, b_buf, _b_step, b_size, elem_size );
                        _b = b_buf;
                    }

                    if( dk0 < len )
                        blockMulFunc( _a, _a_step, _b, _b_step, _d, _d_step,
                                      a_bl_size, cv::Size(dj,di), flags );
                    else
                        singleMulFunc( _a, _a_step, _b, _b_step, _c, Cstep,
                                       _d, _d_step, a_bl_size, cv::Size(dj,di), alpha, beta, flags );
                    flags |= 16;
                }

                if( dk0 < len )
                    storeFunc( _c, Cstep, _d, _d_step,
                               dst.data + i*dst.step + j*elem_size,
                               dst.step, cv::Size(dj,di), alpha, beta, flags );
            }
        }
    }

//    if( MatD != &dst )
//        MatD->copyTo(dst);
    }
}

/****************************************************************************************\
*                                          PCA                                           *
\****************************************************************************************/

faceAnalysis::PCA::PCA() {}

faceAnalysis::PCA::PCA(cv::Mat &data, cv::InputArray _mean, int flags, int maxComponents)
{
    operator()(data, _mean, flags, maxComponents);
}

faceAnalysis::PCA::PCA(cv::Mat &data, cv::InputArray _mean, int flags, double retainedVariance)
{
    faceAnalysis::PCA::computeVar(data, _mean, flags, retainedVariance);
}

faceAnalysis::PCA& faceAnalysis::PCA::operator()(cv::Mat &_data, cv::InputArray __mean, int flags, int maxComponents)
{
    cv::Mat data = _data, _mean = __mean.getMat();
    int covar_flags = cv::COVAR_SCALE; //cv::COVAR_SCALE;
    int i, len, in_count;
    cv::Size mean_sz;

    CV_Assert( data.channels() == 1 );
    if( flags & CV_PCA_DATA_AS_COL )
    {
        len = data.rows;
        in_count = data.cols;
        covar_flags |= cv::COVAR_COLS;
        mean_sz = cv::Size(1, len);
    }
    else
    {
        len = data.cols;
        in_count = data.rows;
        covar_flags |= cv::COVAR_ROWS;
        mean_sz = cv::Size(len, 1);
    }

    int count = std::min(len, in_count), out_count = count;
    if( maxComponents > 0 )
        out_count = std::min(count, maxComponents);

    // "scrambled" way to compute PCA (when cols(A)>rows(A)):
    // B = A'A; B*x=b*x; C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> c = b, x=A'*y
    if( len <= in_count )
        covar_flags |= cv::COVAR_NORMAL;

    int ctype = std::max(CV_32F, data.depth());
    mean.create( mean_sz, ctype );

    cv::Mat covar( count, count, ctype );

    if( _mean.data )
    {
        CV_Assert( _mean.size() == mean_sz );
        _mean.convertTo(mean, ctype);
        covar_flags |= cv::COVAR_USE_AVG;
    }

    cv::calcCovarMatrix( data, covar, mean, covar_flags, ctype );
    eigen( covar, eigenvalues, eigenvectors );

    if( !(covar_flags & cv::COVAR_NORMAL) )
    {
        // CV_PCA_DATA_AS_ROW: cols(A)>rows(A). x=A'*y -> x'=y'*A
        // CV_PCA_DATA_AS_COL: rows(A)>cols(A). x=A''*y -> x'=y'*A'
        cv::Mat tmp_data, tmp_mean, notConst = mean;
        faceAnalysis::repeat(notConst, data.rows/mean.rows, data.cols/mean.cols, tmp_mean);
        if( data.type() != ctype || tmp_mean.data == mean.data )
        {
            data.convertTo( tmp_data, ctype );
            subtract( tmp_data, tmp_mean, tmp_data );
        }
        else
        {
            subtract( data, tmp_mean, tmp_mean );
            tmp_data = tmp_mean;
        }

        cv::Mat evects1(count, len, ctype);
        faceAnalysis::gemm(
                    eigenvectors,
                    tmp_data,
                    1,
                    cv::Mat(),
                    0,
                    evects1,
                    (flags & CV_PCA_DATA_AS_COL) ? CV_GEMM_B_T : 0
                    );

        eigenvectors = evects1;

        // normalize eigenvectors
        for( i = 0; i < out_count; i++ )
        {
            cv::Mat vec = eigenvectors.row(i);
            normalize(vec, vec);
        }
    }

    if( count > out_count )
    {
        // use clone() to physically copy the data and thus deallocate the original cv::Matrices
        eigenvalues = eigenvalues.rowRange(0,out_count).clone();
        eigenvectors = eigenvectors.rowRange(0,out_count).clone();
    }
    return *this;
}

faceAnalysis::PCA& faceAnalysis::PCA::computeVar(cv::Mat &data, cv::InputArray __mean, int flags, double retainedVariance)
{
    cv::Mat _mean = __mean.getMat();
    int covar_flags = cv::COVAR_SCALE;
    int i, len, in_count;
    cv::Size mean_sz;

    CV_Assert( data.channels() == 1 );
    if( flags & CV_PCA_DATA_AS_COL )
    {
        len = data.rows;
        in_count = data.cols;
        covar_flags |= cv::COVAR_COLS; //cv::COVAR_COLS;
        mean_sz = cv::Size(1, len);
    }
    else
    {
        len = data.cols;
        in_count = data.rows;
        covar_flags |= cv::COVAR_ROWS;
        mean_sz = cv::Size(len, 1);
    }

    CV_Assert( retainedVariance > 0 && retainedVariance <= 1 );

    int count = std::min(len, in_count);

    // "scrambled" way to compute PCA (when cols(A)>rows(A)):
    // B = A'A; B*x=b*x; C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> c = b, x=A'*y
    if( len <= in_count )
        covar_flags |= cv::COVAR_NORMAL;

    int ctype = std::max(CV_32F, data.depth());
    mean.create( mean_sz, ctype );

    cv::Mat covar( count, count, ctype );

    if( _mean.data )
    {
        CV_Assert( _mean.size() == mean_sz );
        _mean.convertTo(mean, ctype);
    }
 std::cout << "tick " << std::endl;

    faceAnalysis::calcCovarMatrix( data, covar, mean, covar_flags, ctype );
std::cout << "tock " << std::endl;

    cv::eigen( covar, eigenvalues, eigenvectors );
std::cout << "tick " << std::endl;

    if( !(covar_flags & cv::COVAR_NORMAL) )
    {
        // CV_PCA_DATA_AS_ROW: cols(A)>rows(A). x=A'*y -> x'=y'*A
          // CV_PCA_DATA_AS_COL: rows(A)>cols(A). x=A''*y -> x'=y'*A'
        cv::Mat tmp_data, tmp_mean;
        faceAnalysis::repeat(mean, data.rows/mean.rows, data.cols/mean.cols, tmp_mean);
        if( data.type() != ctype || tmp_mean.data == mean.data )
        {
            data.convertTo( tmp_data, ctype );
            subtract( tmp_data, tmp_mean, tmp_data );
        }
        else
        {
            subtract( data, tmp_mean, tmp_mean );
            tmp_data = tmp_mean;
        }

        cv::Mat evects1(count, len, ctype);
        faceAnalysis::gemm( eigenvectors, tmp_data, 1, cv::Mat(), 0, evects1,
            (flags & CV_PCA_DATA_AS_COL) ? CV_GEMM_B_T : 0);
        eigenvectors = evects1;

        // normalize all eigenvectors
        for( i = 0; i < eigenvectors.rows; i++ )
        {
            cv::Mat vec = eigenvectors.row(i);
            normalize(vec, vec);
        }
    }
    // compute the cumulative energy content for each eigenvector
    int L;
    if (ctype == CV_32F)
        L = faceAnalysis::computeCumulativeEnergy<float>(eigenvalues, retainedVariance);
    else
        L = faceAnalysis::computeCumulativeEnergy<double>(eigenvalues, retainedVariance);

    // use clone() to physically copy the data and thus deallocate the original cv::Matrices
    eigenvalues = eigenvalues.rowRange(0,L).clone();
    eigenvectors = eigenvectors.rowRange(0,L).clone();

    return *this;
}

void faceAnalysis::PCA::project(cv::Mat &data, cv::Mat &result) const
{
    CV_Assert( mean.data && eigenvectors.data &&
        ((mean.rows == 1 && mean.cols == data.cols) || (mean.cols == 1 && mean.rows == data.rows)));
std::cout << "tick project" << std::endl;
    cv::Mat tmp_data, tmp_mean, notConst = mean;
    faceAnalysis::repeat(notConst, data.rows/mean.rows, data.cols/mean.cols, tmp_mean);
std::cout << "tick " << std::endl;
    int ctype = mean.type();
    if( data.type() != ctype || tmp_mean.data == mean.data )
    {
std::cout << "tack " << std::endl;
        data.convertTo( tmp_data, ctype );
        subtract( tmp_data, tmp_mean, tmp_data );
    }
    else
    {
std::cout << "tick " << std::endl;
        subtract( data, tmp_mean, tmp_mean );
std::cout << "tick " << std::endl;
        tmp_data = tmp_mean;
    }
    cv::Mat nonConst = eigenvectors;
    if( mean.rows == 1 ) {
std::cout << "tick " << std::endl;
        faceAnalysis::gemm( tmp_data, nonConst, 1, cv::Mat(), 0, result, cv::GEMM_2_T );

std::cout << "tick " << std::endl;
    } else
        faceAnalysis::gemm( nonConst, tmp_data, 1, cv::Mat(), 0, result, 0 );
}

void faceAnalysis::calcCovarMatrix(
        cv::Mat &data,
        cv::OutputArray _covar,
        cv::InputOutputArray _mean,
        int flags,
        int ctype
        ) {

std::cout << "tuck " << std::endl;
    cv::Mat mean;
    CV_Assert( ((flags & cv::COVAR_ROWS) != 0) ^ ((flags & cv::COVAR_COLS) != 0) );
    bool takeRows = (flags & cv::COVAR_ROWS) != 0;
    int type = data.type();
    int nsamples = takeRows ? data.rows : data.cols;
    CV_Assert( nsamples > 0 );
    cv::Size size = takeRows ? cv::Size(data.cols, 1) : cv::Size(1, data.rows);

    if( (flags & cv::COVAR_USE_AVG) != 0 )    {
        mean = _mean.getMat();
        ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), mean.depth()), CV_32F);
        CV_Assert( mean.size() == size );
        if( mean.type() != ctype )
        {
            _mean.create(mean.size(), ctype);
            cv::Mat tmp = _mean.getMat();
            mean.convertTo(tmp, ctype);
            mean = tmp;
        }
    }
    else
    {
        ctype = std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), CV_32F);

        cv::reduce(
                    data,
                    _mean,
                    takeRows ? 0 : 1,
                    CV_REDUCE_AVG,
                    ctype
                    );
        mean = _mean.getMat();
    }

std::cout << "covar=" << _covar.size() << " data=" << data.size() << std::endl;

    faceAnalysis::mulTransposed(
                data,
                _covar,
                ((flags & cv::COVAR_NORMAL) == 0) ^ takeRows,
                mean,
                (flags & cv::COVAR_SCALE) != 0 ? 1./nsamples : 1,
                ctype
                );
}

        //eig. oben als erstes:
        //    if(_src.kind() == cv::_InputArray::STD_VECTOR_MAT)
        //    {
        //        std::vector<cv::Mat> src;
        //        _src.getMatVector(src);

        //        CV_Assert( src.size() > 0 );

        //        cv::Size size = src[0].size();
        //        int type = src[0].type();

        //        ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), _mean.depth()), CV_32F);

        //        cv::Mat _data(static_cast<int>(src.size()), size.area(), type);

        //        int i = 0;
        //        for(std::vector<cv::Mat>::iterator each = src.begin(); each != src.end(); each++, i++ )
        //        {
        //            CV_Assert( (*each).size() == size && (*each).type() == type );
        //            cv::Mat dataRow(size.height, size.width, type, _data.ptr(i));
        //            (*each).copyTo(dataRow);
        //        }

        //        Mat mean;
        //        if( (flags & cv::COVAR_USE_AVG) != 0 )
        //        {
        //            CV_Assert( _mean.size() == size );

        //            if( mean.type() != ctype )
        //            {
        //                mean = _mean.getMat();
        //                _mean.create(mean.size(), ctype);
        //                Mat tmp = _mean.getMat();
        //                mean.convertTo(tmp, ctype);
        //                mean = tmp;
        //            }

        //            mean = _mean.getMat().reshape(1, 1);
        //        }

        //        faceAnalysis::calcCovarMatrix(
        //                    _data,
        //                    _covar,
        //                    mean,
        //                    (flags & ~(CV_COVAR_ROWS|CV_COVAR_COLS)) | CV_COVAR_ROWS,
        //                    ctype
        //                    );

        //        if( (flags & cv::COVAR_USE_AVG) == 0 )
        //        {
        //            mean = mean.reshape(1, size.height);
        //            mean.copyTo(_mean);
        //        }
        //        return;
        //    }

void faceAnalysis::mulTransposed(
        cv::Mat &src,
        cv::OutputArray _dst,
        bool ata,
        cv::InputArray _delta,
        double scale,
        int dtype
        ) {

 std::cout << "tick mulTrans" << std::endl;
    cv::Mat delta = _delta.getMat();
    const int gemm_level = 100; // boundary above which GEMM is faster.
    int stype = src.type();
    dtype = std::max(std::max(CV_MAT_DEPTH(dtype >= 0 ? dtype : stype), delta.depth()), CV_32F);
    CV_Assert( src.channels() == 1 );

    if( delta.data )
    {
        CV_Assert( delta.channels() == 1 &&
            (delta.rows == src.rows || delta.rows == 1) &&
            (delta.cols == src.cols || delta.cols == 1));
        if( delta.type() != dtype )
            delta.convertTo(delta, dtype);
    }

    int dsize = ata ? src.cols : src.rows;
    _dst.create( dsize, dsize, dtype );
    cv::Mat dst = _dst.getMat();

    if( src.data == dst.data || (stype == dtype &&
        (dst.cols >= gemm_level && dst.rows >= gemm_level &&
         src.cols >= gemm_level && src.rows >= gemm_level)))
    {
std::cout << "tueck" << std::endl;

        cv::Mat src2;
std::cout << "tick" << std::endl;

        cv::Mat* tsrc = &src;
        if( delta.data )
        {
            if( delta.size() == src.size() ) {
                subtract( src, delta, src2 );
             } else {

std::cout << "tiiick" << std::endl;
                faceAnalysis::repeat(delta, src.rows/delta.rows, src.cols/delta.cols, src2);
    std::cout << "tiiick" << std::endl;
                subtract( src, src2, src2 );
std::cout << "tiiick" << std::endl;

            }
            tsrc = &src2;
        }
        faceAnalysis::gemm( *tsrc, *tsrc, scale, cv::Mat(), 0, dst, ata ? cv::GEMM_1_T : cv::GEMM_2_T );
    }
        else
        {
        faceAnalysis::MulTransposedFunc func = 0;
        if(stype == CV_8U && dtype == CV_32F)
        {
            if(ata)
                func = faceAnalysis::MulTransposedR<uchar,float>;
            else
                func = faceAnalysis::MulTransposedL<uchar,float>;
        }
        else if(stype == CV_8U && dtype == CV_64F)
        {
            if(ata)
                func = faceAnalysis::MulTransposedR<uchar,double>;
            else
                func = faceAnalysis::MulTransposedL<uchar,double>;
        }
        else if(stype == CV_16U && dtype == CV_32F)
        {
            if(ata)
                func = faceAnalysis::MulTransposedR<ushort,float>;
            else
                func = faceAnalysis::MulTransposedL<ushort,float>;
        }
        else if(stype == CV_16U && dtype == CV_64F)
        {
            if(ata)
                func = faceAnalysis::MulTransposedR<ushort,double>;
            else
                func = faceAnalysis::MulTransposedL<ushort,double>;
        }
        else if(stype == CV_16S && dtype == CV_32F)
        {
            if(ata)
                func =faceAnalysis::MulTransposedR<short,float>;
            else
                func = faceAnalysis::MulTransposedL<short,float>;
        }
        else if(stype == CV_16S && dtype == CV_64F)
        {
            if(ata)
                func = faceAnalysis::MulTransposedR<short,double>;
            else
                func = faceAnalysis::MulTransposedL<short,double>;
        }
        else if(stype == CV_32F && dtype == CV_32F)
       {
           if(ata)
               func = faceAnalysis::MulTransposedR<float,float>;
           else
               func = faceAnalysis::MulTransposedL<float,float>;
       }
       else if(stype == CV_32F && dtype == CV_64F)
       {
           if(ata)
               func = faceAnalysis::MulTransposedR<float,double>;
           else
               func = faceAnalysis::MulTransposedL<float,double>;
       }
       else if(stype == CV_64F && dtype == CV_64F)
       {
           if(ata)
               func = faceAnalysis::MulTransposedR<double,double>;
           else
               func = faceAnalysis::MulTransposedL<double,double>;
       }
       if( !func )
           CV_Error( CV_StsUnsupportedFormat, "" );

       func( src, dst, delta, scale );
       completeSymm( dst, false );
   }
}

void faceAnalysis::repeat(cv::Mat &src, int ny, int nx, cv::Mat &dst)
{

std::cout << "tick repeat" << std::endl;
    CV_Assert( src.dims <= 2 );
    CV_Assert( ny > 0 && nx > 0 );
std::cout << "create" << src.rows*ny << "x" << src.cols*nx << std::endl;

    dst.create(src.rows*ny, src.cols*nx, src.type());
std::cout << "tick " << std::endl;
    cv::Size ssize = src.size(), dsize = dst.size();
    int esz = (int)src.elemSize();
    int x, y;
    ssize.width *= esz; dsize.width *= esz;
std::cout << ssize.width << " d=" << dsize.width << std::endl;
    for( y = 0; y < ssize.height; y++ )
    {
        for( x = 0; x < dsize.width; x += ssize.width )
            memcpy( dst.data + y*dst.step + x, src.data + y*src.step, ssize.width );
    }

    for( ; y < dsize.height; y++ )
        memcpy( dst.data + y*dst.step, dst.data + (y - ssize.height)*dst.step, dsize.width );
}

//void faceAnalysis::reduce(
//        cv::Mat &src,
//        cv::OutputArray _dst,
//        int dim,
//        int op,
//        int dtype
//        ) {

//    CV_Assert( src.dims <= 2 );
//    int op0 = op;
//    int stype = src.type(), sdepth = src.depth(), cn = src.channels();
//    if( dtype < 0 )
//        dtype = _dst.fixedType() ? _dst.type() : stype;
//    int ddepth = CV_MAT_DEPTH(dtype);

//    _dst.create(dim == 0 ? 1 : src.rows, dim == 0 ? src.cols : 1,
//                CV_MAKETYPE(dtype >= 0 ? dtype : stype, cn));
//    cv::Mat dst = _dst.getMat(), temp = dst;

//    CV_Assert( op == CV_REDUCE_SUM || op == CV_REDUCE_MAX ||
//               op == CV_REDUCE_MIN || op == CV_REDUCE_AVG );
//    CV_Assert( src.channels() == dst.channels() );

//    if( op == CV_REDUCE_AVG )
//    {
//        op = CV_REDUCE_SUM;
//        if( sdepth < CV_32S && ddepth < CV_32S )
//        {
//            temp.create(dst.rows, dst.cols, CV_32SC(cn));
//            ddepth = CV_32S;
//        }
//    }

//    cv::ReduceFunc func = 0;
//    if( dim == 0 )
//    {
//        if( op == CV_REDUCE_SUM )
//        {
//            if(sdepth == CV_8U && ddepth == CV_32S)
//                func = GET_OPTIMIZED(reduceSumR8u32s);
//            else if(sdepth == CV_8U && ddepth == CV_32F)
//                func = reduceSumR8u32f;
//            else if(sdepth == CV_8U && ddepth == CV_64F)
//                func = reduceSumR8u64f;
//            else if(sdepth == CV_16U && ddepth == CV_32F)
//                func = reduceSumR16u32f;
//            else if(sdepth == CV_16U && ddepth == CV_64F)
//                func = reduceSumR16u64f;
//            else if(sdepth == CV_16S && ddepth == CV_32F)
//                func = reduceSumR16s32f;
//            else if(sdepth == CV_16S && ddepth == CV_64F)
//                func = reduceSumR16s64f;
//            else if(sdepth == CV_32F && ddepth == CV_32F)
//                func = GET_OPTIMIZED(reduceSumR32f32f);
//            else if(sdepth == CV_32F && ddepth == CV_64F)
//                func = reduceSumR32f64f;
//            else if(sdepth == CV_64F && ddepth == CV_64F)
//                func = reduceSumR64f64f;
//        }
//        else if(op == CV_REDUCE_MAX)
//        {
//            if(sdepth == CV_8U && ddepth == CV_8U)
//                func = GET_OPTIMIZED(reduceMaxR8u);
//            else if(sdepth == CV_16U && ddepth == CV_16U)
//                func = reduceMaxR16u;
//            else if(sdepth == CV_16S && ddepth == CV_16S)
//                func = reduceMaxR16s;
//            else if(sdepth == CV_32F && ddepth == CV_32F)
//                func = GET_OPTIMIZED(reduceMaxR32f);
//            else if(sdepth == CV_64F && ddepth == CV_64F)
//                func = reduceMaxR64f;
//        }
//        else if(op == CV_REDUCE_MIN)
//        {
//            if(sdepth == CV_8U && ddepth == CV_8U)
//                func = GET_OPTIMIZED(reduceMinR8u);
//            else if(sdepth == CV_16U && ddepth == CV_16U)
//                func = reduceMinR16u;
//            else if(sdepth == CV_16S && ddepth == CV_16S)
//                func = reduceMinR16s;
//            else if(sdepth == CV_32F && ddepth == CV_32F)
//                func = GET_OPTIMIZED(reduceMinR32f);
//            else if(sdepth == CV_64F && ddepth == CV_64F)
//                func = reduceMinR64f;
//        }
//    }
//    else
//    {
//        if(op == CV_REDUCE_SUM)
//        {
//            if(sdepth == CV_8U && ddepth == CV_32S)
//                func = GET_OPTIMIZED(reduceSumC8u32s);
//            else if(sdepth == CV_8U && ddepth == CV_32F)
//                func = reduceSumC8u32f;
//            else if(sdepth == CV_8U && ddepth == CV_64F)
//                func = reduceSumC8u64f;
//            else if(sdepth == CV_16U && ddepth == CV_32F)
//                func = reduceSumC16u32f;
//            else if(sdepth == CV_16U && ddepth == CV_64F)
//                func = reduceSumC16u64f;
//            else if(sdepth == CV_16S && ddepth == CV_32F)
//                func = reduceSumC16s32f;
//            else if(sdepth == CV_16S && ddepth == CV_64F)
//                func = reduceSumC16s64f;
//            else if(sdepth == CV_32F && ddepth == CV_32F)
//                func = GET_OPTIMIZED(reduceSumC32f32f);
//            else if(sdepth == CV_32F && ddepth == CV_64F)
//                func = reduceSumC32f64f;
//            else if(sdepth == CV_64F && ddepth == CV_64F)
//                func = reduceSumC64f64f;
//        }
//        else if(op == CV_REDUCE_MAX)
//        {
//            if(sdepth == CV_8U && ddepth == CV_8U)
//                func = GET_OPTIMIZED(reduceMaxC8u);
//            else if(sdepth == CV_16U && ddepth == CV_16U)
//                func = reduceMaxC16u;
//            else if(sdepth == CV_16S && ddepth == CV_16S)
//                func = reduceMaxC16s;
//            else if(sdepth == CV_32F && ddepth == CV_32F)
//                func = GET_OPTIMIZED(reduceMaxC32f);
//            else if(sdepth == CV_64F && ddepth == CV_64F)
//                func = reduceMaxC64f;
//        }
//        else if(op == CV_REDUCE_MIN)
//        {
//            if(sdepth == CV_8U && ddepth == CV_8U)
//                func = GET_OPTIMIZED(reduceMinC8u);
//            else if(sdepth == CV_16U && ddepth == CV_16U)
//                func = reduceMinC16u;
//            else if(sdepth == CV_16S && ddepth == CV_16S)
//                func = reduceMinC16s;
//            else if(sdepth == CV_32F && ddepth == CV_32F)
//                func = GET_OPTIMIZED(reduceMinC32f);
//            else if(sdepth == CV_64F && ddepth == CV_64F)
//                func = reduceMinC64f;
//        }
//    }

//    if( !func )
//        CV_Error( CV_StsUnsupportedFormat,
//                  "Unsupported combination of input and output array formats" );

//    func( src, temp );

//    if( op0 == CV_REDUCE_AVG )
//        temp.convertTo(dst, dst.type(), 1./(dim == 0 ? src.rows : src.cols));
//}

//cv::Mat faceAnalysis::PCA::project(cv::Mat &data) const
//{
//    cv::Mat result;
//    project(data, result);
//    return result;
//}

//void faceAnalysis::PCA::backProject(cv::InputArray _data, cv::OutputArray result) const
//{
//    cv::Mat data = _data.getMat();
//    CV_Assert( mean.data && eigenvectors.data &&
//        ((mean.rows == 1 && eigenvectors.rows == data.cols) ||
//         (mean.cols == 1 && eigenvectors.rows == data.rows)));

//    cv::Mat tmp_data, tmp_mean;
//    data.convertTo(tmp_data, mean.type());
//    if( mean.rows == 1 )
//    {
//        tmp_mean = repeat(mean, data.rows, 1);
//        faceAnalysis::gemm( tmp_data, eigenvectors, 1, tmp_mean, 1, result, 0 );
//    }
//    else
//    {
//        tmp_mean = repeat(mean, 1, data.cols);
//        faceAnalysis::gemm(eigenvectors , tmp_data, 1, tmp_mean, 1, result, cv::GEMM_1_T );
//    }
//}

//cv::Mat faceAnalysis::PCA::backProject(cv::Mat &data) const
//{
//    cv::Mat result;
//    backProject(data, result);
//    return result;
//}

//void faceAnalysis::PCACompute(cv::InputArray data, cv::InputOutputArray mean,
//                    cv::OutputArray eigenvectors, int maxComponents)
//{
//    PCA pca;
//    pca(data, mean, 0, maxComponents);
//    pca.mean.copyTo(mean);
//    pca.eigenvectors.copyTo(eigenvectors);
//}

//void faceAnalysis::PCAComputeVar(cv::InputArray data, cv::InputOutputArray mean,
//                    cv::OutputArray eigenvectors, double retainedVariance)
//{
//    PCA pca;
//    pca.computeVar(data, mean, 0, retainedVariance);
//    pca.mean.copyTo(mean);
//    pca.eigenvectors.copyTo(eigenvectors);
//}

//void faceAnalysis::PCAProject(cv::InputArray data, cv::InputArray mean,
//                    cv::InputArray eigenvectors, cv::OutputArray result)
//{
//    PCA pca;
//    pca.mean = mean.getMat();
//    pca.eigenvectors = eigenvectors.getMat();
//    pca.project(data, result);
//}

//void faceAnalysis::PCABackProject(cv::InputArray data, cv::InputArray mean,
//                    cv::InputArray eigenvectors, cv::OutputArray result)
//{
//    PCA pca;
//    pca.mean = mean.getMat();
//    pca.eigenvectors = eigenvectors.getMat();
//    pca.backProject(data, result);
//}
