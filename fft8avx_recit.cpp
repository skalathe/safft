#include <bits/stdc++.h>
#include <immintrin.h>

#define _MM256_TRANSPOSE_4x4_PD(row0,row1,row2,row3) \
    do { \
        __m256d __t3, __t2, __t1, __t0; \
        __t0 = _mm256_shuffle_pd((row0),(row1), 0x0); \
        __t2 = _mm256_shuffle_pd((row0),(row1), 0xF); \
        __t1 = _mm256_shuffle_pd((row2),(row3), 0x0); \
        __t3 = _mm256_shuffle_pd((row2),(row3), 0xF); \
        (row0) = _mm256_permute2f128_pd(__t0, __t1, 0x20); \
        (row1) = _mm256_permute2f128_pd(__t2, __t3, 0x20); \
        (row2) = _mm256_permute2f128_pd(__t0, __t1, 0x31); \
        (row3) = _mm256_permute2f128_pd(__t2, __t3, 0x31); \
    } while (0)

#define native_alignment 32
#define cache_fit 1024
#define vector_width 4
#define vector_next 8
#define fft_MAX 32

static const double E4 = std::sin(M_PI_4);

inline void revbin_permute( double * data, uint32_t n ) {

    if ( n <= 2 ) return;

    uint32_t N2   = n >> 1;
    uint32_t r    = 0;
    uint32_t x    = 1;

    while( x < N2 ) {
        // x is odd
        r = r + N2;
        std::swap(data[x*2],data[r*2]);
		std::swap(data[x*2+1],data[r*2+1]);
        x = x + 1;
        // x is even
        for ( uint32_t m = n >> 1; (!((r ^= m) & m)); m >>= 1 );

        if ( r > x ) {
            std::swap(data[x*2],data[r*2]);
			std::swap(data[x*2+1],data[r*2+1]);
            std::swap(data[(n-1-x)*2],data[(n-1-r)*2]);
            std::swap(data[(n-1-x)*2+1],data[(n-1-r)*2+1]);
        }
        x = x + 1;
    }
}

struct fft_plan {

    size_t size;
    size_t depth;
    double * twiddle;

    inline void initialize_twiddles( double *& twiddle, size_t stage_offs, size_t blocks_offs, size_t size )
    {
        double rr, ii;

        sincos( 2 * M_PI * (stage_offs                   ) / size, &ii, &rr);
        twiddle[0]                =  rr;
        twiddle[vector_width]     = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs * 2 ) / size, &ii, &rr);
        twiddle[1]                =  rr;
        twiddle[1 + vector_width] = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs     ) / size, &ii, &rr);
        twiddle[2]                =  rr;
        twiddle[2 + vector_width] = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs * 3 ) / size, &ii, &rr);
        twiddle[3]                =  rr;
        twiddle[3 + vector_width] = -ii;

        twiddle += 2 * vector_width;
    }

    fft_plan ( size_t size ) : size ( size ) {

				size_t twiddle_size = 0;

				for ( size_t s = size; s > 8; s /= 8 ) twiddle_size |= s;

        twiddle = (double*) aligned_alloc( native_alignment, 2 * twiddle_size * sizeof(double) );

        double * twiddleinit = twiddle;

        depth = 1;

        for ( size_t stage_size = size, blocks = 1; stage_size > 8; stage_size >>= 3, blocks <<= 3 ) {

        	depth++;

            for ( size_t n = 0; n < stage_size >> 3; n += vector_width ) {

                initialize_twiddles( twiddleinit, n * blocks    , blocks    , size );
                initialize_twiddles( twiddleinit, n * blocks * 2, blocks * 2, size );
                initialize_twiddles( twiddleinit, n * blocks * 3, blocks * 3, size );
                initialize_twiddles( twiddleinit, n * blocks * 4, blocks * 4, size );
                initialize_twiddles( twiddleinit, n * blocks * 5, blocks * 5, size );
                initialize_twiddles( twiddleinit, n * blocks * 6, blocks * 6, size );
                initialize_twiddles( twiddleinit, n * blocks * 7, blocks * 7, size );
            }
        }
    }

    void fft_free ( void ) {

        free(twiddle);

    }

    inline void butterfly_zero( const double * in, double * data, const double * twiddle, uint32_t stage_size ) {

        const size_t N8 = stage_size >> 2;

        __m256d xmx = _mm256_load_pd( in 	          );
        __m256d xmy = _mm256_load_pd( in          + 4 );
        __m256d re0 = _mm256_unpacklo_pd( xmx, xmy );
        __m256d im0 = _mm256_unpackhi_pd( xmx, xmy );
                xmx = _mm256_load_pd( in + N8         );
                xmy = _mm256_load_pd( in + N8     + 4 );
        __m256d re1 = _mm256_unpacklo_pd( xmx, xmy );
        __m256d im1 = _mm256_unpackhi_pd( xmx, xmy );
                xmx = _mm256_load_pd( in + N8 * 2     );
                xmy = _mm256_load_pd( in + N8 * 2 + 4 );
        __m256d re2 = _mm256_unpacklo_pd( xmx, xmy );
        __m256d im2 = _mm256_unpackhi_pd( xmx, xmy );
                xmx = _mm256_load_pd( in + N8 * 3     );
                xmy = _mm256_load_pd( in + N8 * 3 + 4 );
        __m256d re3 = _mm256_unpacklo_pd( xmx, xmy );
        __m256d im3 = _mm256_unpackhi_pd( xmx, xmy );
                xmx = _mm256_load_pd( in + N8 * 4     );
                xmy = _mm256_load_pd( in + N8 * 4 + 4 );
        __m256d re4 = _mm256_unpacklo_pd( xmx, xmy );
        __m256d im4 = _mm256_unpackhi_pd( xmx, xmy );
                xmx = _mm256_load_pd( in + N8 * 5     );
                xmy = _mm256_load_pd( in + N8 * 5 + 4 );
        __m256d re5 = _mm256_unpacklo_pd( xmx, xmy );
        __m256d im5 = _mm256_unpackhi_pd( xmx, xmy );
                xmx = _mm256_load_pd( in + N8 * 6     );
                xmy = _mm256_load_pd( in + N8 * 6 + 4 );
        __m256d re6 = _mm256_unpacklo_pd( xmx, xmy );
        __m256d im6 = _mm256_unpackhi_pd( xmx, xmy );
                xmx = _mm256_load_pd( in + N8 * 7     );
                xmy = _mm256_load_pd( in + N8 * 7 + 4 );
        __m256d re7 = _mm256_unpacklo_pd( xmx, xmy );
        __m256d im7 = _mm256_unpackhi_pd( xmx, xmy );

        __m256d sum04re = _mm256_add_pd( re0, re4 );
        __m256d sum04im = _mm256_add_pd( im0, im4 );
        __m256d sum15re = _mm256_add_pd( re1, re5 );
        __m256d sum15im = _mm256_add_pd( im1, im5 );
        __m256d sum26re = _mm256_add_pd( re2, re6 );
        __m256d sum26im = _mm256_add_pd( im2, im6 );
        __m256d sum37re = _mm256_add_pd( re3, re7 );
        __m256d sum37im = _mm256_add_pd( im3, im7 );

        __m256d xma, xmb, xE4;

        //0,4
        xma = _mm256_add_pd(sum04re, sum26re);
        xmb = _mm256_add_pd(sum37re, sum15re);

        _mm256_store_pd( data, _mm256_add_pd( xma, xmb ) );
        __m256d w4r = _mm256_sub_pd(xma, xmb);

        xma = _mm256_add_pd(sum04im, sum26im);
        xmb = _mm256_add_pd(sum37im, sum15im);

        _mm256_store_pd( data + 4, _mm256_add_pd( xma, xmb ) );
        __m256d w4i = _mm256_sub_pd( xma, xmb );

        __m256d tw4r = _mm256_load_pd(twiddle + 6 * vector_width );
        __m256d tw4i = _mm256_load_pd(twiddle + 7 * vector_width );

        __m256d tw4or = _mm256_sub_pd( _mm256_mul_pd(w4r, tw4r), _mm256_mul_pd(w4i, tw4i) );
        __m256d tw4oi = _mm256_add_pd( _mm256_mul_pd(w4i, tw4r), _mm256_mul_pd(w4r, tw4i) );

        _mm256_store_pd( data + N8    , tw4or );
        _mm256_store_pd( data + N8 + 4, tw4oi );

        //2,6
        xma = _mm256_sub_pd(sum04re, sum26re);
        xmb = _mm256_sub_pd(sum37im, sum15im);
        __m256d w2r = _mm256_sub_pd(xma, xmb);
        __m256d w6r = _mm256_add_pd(xma, xmb);
        xma = _mm256_sub_pd(sum04im, sum26im);
        xmb = _mm256_sub_pd(sum37re, sum15re);
        __m256d w2i = _mm256_add_pd(xma, xmb);
        __m256d w6i = _mm256_sub_pd(xma, xmb);


        __m256d tw2r = _mm256_load_pd(twiddle + 2  * vector_width );
        __m256d tw2i = _mm256_load_pd(twiddle + 3  * vector_width );
        __m256d tw6r = _mm256_load_pd(twiddle + 10 * vector_width );
        __m256d tw6i = _mm256_load_pd(twiddle + 11 * vector_width );

         __m256d tw2or = _mm256_sub_pd( _mm256_mul_pd(w2r, tw2r), _mm256_mul_pd(w2i, tw2i) );
         __m256d tw2oi = _mm256_add_pd( _mm256_mul_pd(w2i, tw2r), _mm256_mul_pd(w2r, tw2i) );
         __m256d tw6or = _mm256_sub_pd( _mm256_mul_pd(w6r, tw6r), _mm256_mul_pd(w6i, tw6i) );
         __m256d tw6oi = _mm256_add_pd( _mm256_mul_pd(w6i, tw6r), _mm256_mul_pd(w6r, tw6i) );

        _mm256_store_pd( data + N8 * 2    , tw2or );
        _mm256_store_pd( data + N8 * 2 + 4, tw2oi );
        _mm256_store_pd( data + N8 * 3    , tw6or );
        _mm256_store_pd( data + N8 * 3 + 4, tw6oi );

        __m256d diff04re = _mm256_sub_pd( re0, re4 );
        __m256d diff04im = _mm256_sub_pd( im0, im4 );
        __m256d diff15re = _mm256_sub_pd( re1, re5 );
        __m256d diff15im = _mm256_sub_pd( im1, im5 );
        __m256d diff26re = _mm256_sub_pd( re2, re6 );
        __m256d diff26im = _mm256_sub_pd( im2, im6 );
        __m256d diff37re = _mm256_sub_pd( re3, re7 );
        __m256d diff37im = _mm256_sub_pd( im3, im7 );

        //3,7
        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_add_pd(_mm256_add_pd(diff37re, diff37im), _mm256_sub_pd(diff15im, diff15re)) );

        xma = _mm256_sub_pd(diff04re, diff26im);

        __m256d w3r = _mm256_add_pd(xma, xE4);
        __m256d w7r = _mm256_sub_pd(xma, xE4);

        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_sub_pd(_mm256_sub_pd(diff37im, diff37re), _mm256_add_pd(diff15re, diff15im)) );

        xma = _mm256_add_pd(diff04im, diff26re);

        __m256d w3i = _mm256_add_pd(xma, xE4);
        __m256d w7i = _mm256_sub_pd(xma, xE4);

        __m256d tw3r = _mm256_load_pd(twiddle + 4  * vector_width );
        __m256d tw3i = _mm256_load_pd(twiddle + 5  * vector_width );
        __m256d tw7r = _mm256_load_pd(twiddle + 12 * vector_width );
        __m256d tw7i = _mm256_load_pd(twiddle + 13 * vector_width );

        __m256d tw3or = _mm256_sub_pd( _mm256_mul_pd(w3r, tw3r), _mm256_mul_pd(w3i, tw3i) );
        __m256d tw3oi = _mm256_add_pd( _mm256_mul_pd(w3i, tw3r), _mm256_mul_pd(w3r, tw3i) );
        __m256d tw7or = _mm256_sub_pd( _mm256_mul_pd(w7r, tw7r), _mm256_mul_pd(w7i, tw7i) );
        __m256d tw7oi = _mm256_add_pd( _mm256_mul_pd(w7i, tw7r), _mm256_mul_pd(w7r, tw7i) );

        _mm256_store_pd( data + N8 * 6    , tw3or );
        _mm256_store_pd( data + N8 * 6 + 4, tw3oi );
        _mm256_store_pd( data + N8 * 7    , tw7or );
        _mm256_store_pd( data + N8 * 7 + 4, tw7oi );

        //1,5
        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_add_pd(_mm256_sub_pd(diff37im, diff37re), _mm256_add_pd(diff15re, diff15im)) );

        xma = _mm256_add_pd(diff04re, diff26im);

        __m256d w5r = _mm256_sub_pd(xma, xE4);
        __m256d w1r = _mm256_add_pd(xma, xE4);

        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_sub_pd(_mm256_add_pd(diff37re, diff37im), _mm256_sub_pd(diff15im, diff15re)) );

        xma = _mm256_sub_pd(diff04im, diff26re);

        __m256d w5i = _mm256_add_pd(xma, xE4);
        __m256d w1i = _mm256_sub_pd(xma, xE4);

        __m256d tw1r = _mm256_load_pd(twiddle            );
        __m256d tw1i = _mm256_load_pd(twiddle + vector_width    );
        __m256d tw5r = _mm256_load_pd(twiddle + 8 * vector_width);
        __m256d tw5i = _mm256_load_pd(twiddle + 9 * vector_width);

        __m256d tw1or = _mm256_sub_pd( _mm256_mul_pd(w1r, tw1r), _mm256_mul_pd(w1i, tw1i) );
        __m256d tw1oi = _mm256_add_pd( _mm256_mul_pd(w1i, tw1r), _mm256_mul_pd(w1r, tw1i) );
        __m256d tw5or = _mm256_sub_pd( _mm256_mul_pd(w5r, tw5r), _mm256_mul_pd(w5i, tw5i) );
        __m256d tw5oi = _mm256_add_pd( _mm256_mul_pd(w5i, tw5r), _mm256_mul_pd(w5r, tw5i) );

        _mm256_store_pd( data + N8 * 4    , tw1or );
        _mm256_store_pd( data + N8 * 4 + 4, tw1oi );
        _mm256_store_pd( data + N8 * 5    , tw5or );
        _mm256_store_pd( data + N8 * 5 + 4, tw5oi );

    }

    inline void butterfly( double * data, const double * twiddle, uint32_t stage_size ) {

        const size_t N8 = stage_size >> 2;

        __m256d re0 = _mm256_load_pd(data              );
        __m256d im0 = _mm256_load_pd(data          + 4 );
        __m256d re1 = _mm256_load_pd(data + N8         );
        __m256d im1 = _mm256_load_pd(data + N8     + 4 );
        __m256d re2 = _mm256_load_pd(data + N8 * 2     );
        __m256d im2 = _mm256_load_pd(data + N8 * 2 + 4 );
        __m256d re3 = _mm256_load_pd(data + N8 * 3     );
        __m256d im3 = _mm256_load_pd(data + N8 * 3 + 4 );
        __m256d re4 = _mm256_load_pd(data + N8 * 4     );
        __m256d im4 = _mm256_load_pd(data + N8 * 4 + 4 );
        __m256d re5 = _mm256_load_pd(data + N8 * 5     );
        __m256d im5 = _mm256_load_pd(data + N8 * 5 + 4 );
        __m256d re6 = _mm256_load_pd(data + N8 * 6     );
        __m256d im6 = _mm256_load_pd(data + N8 * 6 + 4 );
        __m256d re7 = _mm256_load_pd(data + N8 * 7     );
        __m256d im7 = _mm256_load_pd(data + N8 * 7 + 4 );

        __m256d sum04re = _mm256_add_pd( re0, re4 );
        __m256d sum04im = _mm256_add_pd( im0, im4 );
        __m256d sum15re = _mm256_add_pd( re1, re5 );
        __m256d sum15im = _mm256_add_pd( im1, im5 );
        __m256d sum26re = _mm256_add_pd( re2, re6 );
        __m256d sum26im = _mm256_add_pd( im2, im6 );
        __m256d sum37re = _mm256_add_pd( re3, re7 );
        __m256d sum37im = _mm256_add_pd( im3, im7 );

        __m256d xma, xmb, xE4;

        //0,4
        xma = _mm256_add_pd(sum04re, sum26re);
        xmb = _mm256_add_pd(sum37re, sum15re);

        _mm256_store_pd( data, _mm256_add_pd( xma, xmb ) );
        __m256d w4r = _mm256_sub_pd(xma, xmb);

        xma = _mm256_add_pd(sum04im, sum26im);
        xmb = _mm256_add_pd(sum37im, sum15im);

        _mm256_store_pd( data + 4, _mm256_add_pd( xma, xmb ) );
        __m256d w4i = _mm256_sub_pd( xma, xmb );

        __m256d tw4r = _mm256_load_pd(twiddle + 6 * vector_width );
        __m256d tw4i = _mm256_load_pd(twiddle + 7 * vector_width );

        __m256d tw4or = _mm256_sub_pd( _mm256_mul_pd(w4r, tw4r), _mm256_mul_pd(w4i, tw4i) );
        __m256d tw4oi = _mm256_add_pd( _mm256_mul_pd(w4i, tw4r), _mm256_mul_pd(w4r, tw4i) );

        _mm256_store_pd( data + N8    , tw4or );
        _mm256_store_pd( data + N8 + 4, tw4oi );

        //2,6
        xma = _mm256_sub_pd(sum04re, sum26re);
        xmb = _mm256_sub_pd(sum37im, sum15im);
        __m256d w2r = _mm256_sub_pd(xma, xmb);
        __m256d w6r = _mm256_add_pd(xma, xmb);
        xma = _mm256_sub_pd(sum04im, sum26im);
        xmb = _mm256_sub_pd(sum37re, sum15re);
        __m256d w2i = _mm256_add_pd(xma, xmb);
        __m256d w6i = _mm256_sub_pd(xma, xmb);


        __m256d tw2r = _mm256_load_pd(twiddle + 2  * vector_width );
        __m256d tw2i = _mm256_load_pd(twiddle + 3  * vector_width );
        __m256d tw6r = _mm256_load_pd(twiddle + 10 * vector_width );
        __m256d tw6i = _mm256_load_pd(twiddle + 11 * vector_width );

         __m256d tw2or = _mm256_sub_pd( _mm256_mul_pd(w2r, tw2r), _mm256_mul_pd(w2i, tw2i) );
         __m256d tw2oi = _mm256_add_pd( _mm256_mul_pd(w2i, tw2r), _mm256_mul_pd(w2r, tw2i) );
         __m256d tw6or = _mm256_sub_pd( _mm256_mul_pd(w6r, tw6r), _mm256_mul_pd(w6i, tw6i) );
         __m256d tw6oi = _mm256_add_pd( _mm256_mul_pd(w6i, tw6r), _mm256_mul_pd(w6r, tw6i) );

        _mm256_store_pd( data + N8 * 2    , tw2or );
        _mm256_store_pd( data + N8 * 2 + 4, tw2oi );
        _mm256_store_pd( data + N8 * 3    , tw6or );
        _mm256_store_pd( data + N8 * 3 + 4, tw6oi );

        __m256d diff04re = _mm256_sub_pd( re0, re4 );
        __m256d diff04im = _mm256_sub_pd( im0, im4 );
        __m256d diff15re = _mm256_sub_pd( re1, re5 );
        __m256d diff15im = _mm256_sub_pd( im1, im5 );
        __m256d diff26re = _mm256_sub_pd( re2, re6 );
        __m256d diff26im = _mm256_sub_pd( im2, im6 );
        __m256d diff37re = _mm256_sub_pd( re3, re7 );
        __m256d diff37im = _mm256_sub_pd( im3, im7 );

        //3,7
        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_add_pd(_mm256_add_pd(diff37re, diff37im), _mm256_sub_pd(diff15im, diff15re)) );

        xma = _mm256_sub_pd(diff04re, diff26im);

        __m256d w3r = _mm256_add_pd(xma, xE4);
        __m256d w7r = _mm256_sub_pd(xma, xE4);

        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_sub_pd(_mm256_sub_pd(diff37im, diff37re), _mm256_add_pd(diff15re, diff15im)) );

        xma = _mm256_add_pd(diff04im, diff26re);

        __m256d w3i = _mm256_add_pd(xma, xE4);
        __m256d w7i = _mm256_sub_pd(xma, xE4);

        __m256d tw3r = _mm256_load_pd(twiddle + 4  * vector_width );
        __m256d tw3i = _mm256_load_pd(twiddle + 5  * vector_width );
        __m256d tw7r = _mm256_load_pd(twiddle + 12 * vector_width );
        __m256d tw7i = _mm256_load_pd(twiddle + 13 * vector_width );

        __m256d tw3or = _mm256_sub_pd( _mm256_mul_pd(w3r, tw3r), _mm256_mul_pd(w3i, tw3i) );
        __m256d tw3oi = _mm256_add_pd( _mm256_mul_pd(w3i, tw3r), _mm256_mul_pd(w3r, tw3i) );
        __m256d tw7or = _mm256_sub_pd( _mm256_mul_pd(w7r, tw7r), _mm256_mul_pd(w7i, tw7i) );
        __m256d tw7oi = _mm256_add_pd( _mm256_mul_pd(w7i, tw7r), _mm256_mul_pd(w7r, tw7i) );

        _mm256_store_pd( data + N8 * 6    , tw3or );
        _mm256_store_pd( data + N8 * 6 + 4, tw3oi );
        _mm256_store_pd( data + N8 * 7    , tw7or );
        _mm256_store_pd( data + N8 * 7 + 4, tw7oi );

        //1,5
        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_add_pd(_mm256_sub_pd(diff37im, diff37re), _mm256_add_pd(diff15re, diff15im)) );

        xma = _mm256_add_pd(diff04re, diff26im);

        __m256d w5r = _mm256_sub_pd(xma, xE4);
        __m256d w1r = _mm256_add_pd(xma, xE4);

        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_sub_pd(_mm256_add_pd(diff37re, diff37im), _mm256_sub_pd(diff15im, diff15re)) );

        xma = _mm256_sub_pd(diff04im, diff26re);

        __m256d w5i = _mm256_add_pd(xma, xE4);
        __m256d w1i = _mm256_sub_pd(xma, xE4);

        __m256d tw1r = _mm256_load_pd(twiddle            );
        __m256d tw1i = _mm256_load_pd(twiddle + vector_width    );
        __m256d tw5r = _mm256_load_pd(twiddle + 8 * vector_width);
        __m256d tw5i = _mm256_load_pd(twiddle + 9 * vector_width);

        __m256d tw1or = _mm256_sub_pd( _mm256_mul_pd(w1r, tw1r), _mm256_mul_pd(w1i, tw1i) );
        __m256d tw1oi = _mm256_add_pd( _mm256_mul_pd(w1i, tw1r), _mm256_mul_pd(w1r, tw1i) );
        __m256d tw5or = _mm256_sub_pd( _mm256_mul_pd(w5r, tw5r), _mm256_mul_pd(w5i, tw5i) );
        __m256d tw5oi = _mm256_add_pd( _mm256_mul_pd(w5i, tw5r), _mm256_mul_pd(w5r, tw5i) );

        _mm256_store_pd( data + N8 * 4    , tw1or );
        _mm256_store_pd( data + N8 * 4 + 4, tw1oi );
        _mm256_store_pd( data + N8 * 5    , tw5or );
        _mm256_store_pd( data + N8 * 5 + 4, tw5oi );

    }

    inline void butterfly_final( double * data ) {

        __m256d re0 = _mm256_load_pd(data     );
        __m256d im0 = _mm256_load_pd(data + 4 );
        __m256d re4 = _mm256_load_pd(data + 8 );
        __m256d im4 = _mm256_load_pd(data + 12);
        __m256d re1 = _mm256_load_pd(data + 16);
        __m256d im1 = _mm256_load_pd(data + 20);
        __m256d re5 = _mm256_load_pd(data + 24);
        __m256d im5 = _mm256_load_pd(data + 28);
        __m256d re2 = _mm256_load_pd(data + 32);
        __m256d im2 = _mm256_load_pd(data + 36);
        __m256d re6 = _mm256_load_pd(data + 40);
        __m256d im6 = _mm256_load_pd(data + 44);
        __m256d re3 = _mm256_load_pd(data + 48);
        __m256d im3 = _mm256_load_pd(data + 52);
        __m256d re7 = _mm256_load_pd(data + 56);
        __m256d im7 = _mm256_load_pd(data + 60);

        _MM256_TRANSPOSE_4x4_PD(re0, re1, re2, re3);
        _MM256_TRANSPOSE_4x4_PD(re4, re5, re6, re7);
        _MM256_TRANSPOSE_4x4_PD(im0, im1, im2, im3);
        _MM256_TRANSPOSE_4x4_PD(im4, im5, im6, im7);

        __m256d sum04re = _mm256_add_pd( re0, re4 );    //swap 1 & 2 and also 5 & 6 to negate unpack changes
        __m256d sum04im = _mm256_add_pd( im0, im4 );
        __m256d sum15re = _mm256_add_pd( re2, re6 );
        __m256d sum15im = _mm256_add_pd( im2, im6 );
        __m256d sum26re = _mm256_add_pd( re1, re5 );
        __m256d sum26im = _mm256_add_pd( im1, im5 );
        __m256d sum37re = _mm256_add_pd( re3, re7 );
        __m256d sum37im = _mm256_add_pd( im3, im7 );

        __m256d xma, xmb, xE4;

        //0,4
        xma = _mm256_add_pd(sum04re, sum26re);
        xmb = _mm256_add_pd(sum37re, sum15re);

        __m256d w0r = _mm256_add_pd( xma, xmb );
        __m256d w2r = _mm256_sub_pd( xma, xmb );

        xma = _mm256_add_pd(sum04im, sum26im);
        xmb = _mm256_add_pd(sum37im, sum15im);

        __m256d w0i = _mm256_add_pd( xma, xmb );
        __m256d w2i = _mm256_sub_pd( xma, xmb );

        //2,6
        xma = _mm256_sub_pd(sum04re, sum26re);
        xmb = _mm256_sub_pd(sum37im, sum15im);
        __m256d w1r = _mm256_sub_pd(xma, xmb);
        __m256d w3r = _mm256_add_pd(xma, xmb);
        xma = _mm256_sub_pd(sum04im, sum26im);
        xmb = _mm256_sub_pd(sum37re, sum15re);
        __m256d w1i = _mm256_add_pd(xma, xmb);
        __m256d w3i = _mm256_sub_pd(xma, xmb);

        __m256d diff04re = _mm256_sub_pd( re0, re4 );     //swap 1 & 2 and also 5 & 6 to negate unpack changes
        __m256d diff04im = _mm256_sub_pd( im0, im4 );
        __m256d diff15re = _mm256_sub_pd( re2, re6 );
        __m256d diff15im = _mm256_sub_pd( im2, im6 );
        __m256d diff26re = _mm256_sub_pd( re1, re5 );
        __m256d diff26im = _mm256_sub_pd( im1, im5 );
        __m256d diff37re = _mm256_sub_pd( re3, re7 );
        __m256d diff37im = _mm256_sub_pd( im3, im7 );

        //3,7
        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_add_pd(_mm256_add_pd(diff37re, diff37im), _mm256_sub_pd(diff15im, diff15re)) );

        xma = _mm256_sub_pd(diff04re, diff26im);

        __m256d w5r = _mm256_add_pd(xma, xE4);
        __m256d w7r = _mm256_sub_pd(xma, xE4);

        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_sub_pd(_mm256_sub_pd(diff37im, diff37re), _mm256_add_pd(diff15re, diff15im)) );

        xma = _mm256_add_pd(diff04im, diff26re);

        __m256d w5i = _mm256_add_pd(xma, xE4);
        __m256d w7i = _mm256_sub_pd(xma, xE4);

        //1,5
        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_add_pd(_mm256_sub_pd(diff37im, diff37re), _mm256_add_pd(diff15re, diff15im)) );

        xma = _mm256_add_pd(diff04re, diff26im);

        __m256d w6r = _mm256_sub_pd(xma, xE4);
        __m256d w4r = _mm256_add_pd(xma, xE4);

        xE4 = _mm256_mul_pd( _mm256_set1_pd(E4), _mm256_sub_pd(_mm256_add_pd(diff37re, diff37im), _mm256_sub_pd(diff15im, diff15re)) );

        xma = _mm256_sub_pd(diff04im, diff26re);

        __m256d w6i = _mm256_add_pd(xma, xE4);
        __m256d w4i = _mm256_sub_pd(xma, xE4);

        _MM256_TRANSPOSE_4x4_PD(w0r, w1r, w2r, w3r);
        _MM256_TRANSPOSE_4x4_PD(w4r, w5r, w6r, w7r);
        _MM256_TRANSPOSE_4x4_PD(w0i, w1i, w2i, w3i);
        _MM256_TRANSPOSE_4x4_PD(w4i, w5i, w6i, w7i);

        _mm256_store_pd( data     , _mm256_unpacklo_pd(w0r,w0i));
        _mm256_store_pd( data + 4 , _mm256_unpackhi_pd(w0r,w0i));
        _mm256_store_pd( data + 8 , _mm256_unpacklo_pd(w4r,w4i));
        _mm256_store_pd( data + 12, _mm256_unpackhi_pd(w4r,w4i));
        _mm256_store_pd( data + 16, _mm256_unpacklo_pd(w1r,w1i));
        _mm256_store_pd( data + 20, _mm256_unpackhi_pd(w1r,w1i));
        _mm256_store_pd( data + 24, _mm256_unpacklo_pd(w5r,w5i));
        _mm256_store_pd( data + 28, _mm256_unpackhi_pd(w5r,w5i));
        _mm256_store_pd( data + 32, _mm256_unpacklo_pd(w2r,w2i));
        _mm256_store_pd( data + 36, _mm256_unpackhi_pd(w2r,w2i));
        _mm256_store_pd( data + 40, _mm256_unpacklo_pd(w6r,w6i));
        _mm256_store_pd( data + 44, _mm256_unpackhi_pd(w6r,w6i));
        _mm256_store_pd( data + 48, _mm256_unpacklo_pd(w3r,w3i));
        _mm256_store_pd( data + 52, _mm256_unpackhi_pd(w3r,w3i));
        _mm256_store_pd( data + 56, _mm256_unpacklo_pd(w7r,w7i));
        _mm256_store_pd( data + 60, _mm256_unpackhi_pd(w7r,w7i));

    }

	// fixed stack for ffts up to 2^32 (can be increased)
	size_t 	 stack [fft_MAX][2];
	double * stack_[fft_MAX][2];
	#define SIZE 0
	#define ITER 1
	#define DATA 0
	#define TWID 1
	#define TOP(ITEM)  stack[st_ptr-1][ITEM]
	#define TOP_(ITEM) stack_[st_ptr-1][ITEM]
	uint32_t st_ptr = 0;

	inline void st_push( size_t size, double * data, double * twiddle ) {
		stack [st_ptr]  [SIZE] = size;
		stack [st_ptr]  [ITER] = 0;
		stack_[st_ptr]  [DATA] = data;
		stack_[st_ptr++][TWID] = twiddle;
	}

	inline void st_pop( void ) {
		st_ptr--;
	}

	bool st_next( void ) {
		if ( TOP(ITER)++ < 8 ) return true;
		return false;
	}

	inline void fft_recursive( double * data, double * twiddle_data ) {

		st_push( size >> 3, data, twiddle_data );

		while ( st_ptr ) { // stack not empty

			if ( st_next() ) {

				if ( TOP(SIZE) == 512 ) {

					double * subdata = TOP_(DATA);
					double * subtwiddle = TOP_(TWID);

					// 512 unroll

						for ( size_t n = 0; n < 64; n += vector_width ) {

							butterfly( subdata, subtwiddle + n * 14, 512 );
							subdata += vector_next;
						}

					subtwiddle += 896;

					// 64 unroll

					subdata = TOP_(DATA);

					for ( size_t i = 0; i < 512; i += 64 ) {

						for ( size_t n = 0; n < 8; n += vector_width ) {

							butterfly( subdata, subtwiddle + n * 14, 64 );
							subdata += vector_next;
						}
						subdata += 112;
					}

					// 8 unroll

					subdata = TOP_(DATA);

					for ( size_t n = 0; n < 512; n += 32 ) {

						butterfly_final( subdata );
						subdata += 64;
					}

					TOP_(DATA) += 1024;

				} else {

					const size_t N8 = TOP(SIZE) >> 3;
					double * safe = TOP_(DATA);

					for ( size_t n = 0; n < N8; n += vector_width ) {

						butterfly( TOP_(DATA), TOP_(TWID) + n * 14, TOP(SIZE) );
						TOP_(DATA) += vector_next;
					}
					TOP_(DATA) += 14 * N8;

					st_push( N8, safe, TOP_(TWID) + 14 * N8 );
				}

			} else st_pop();
		}
	}

    void fft( const double * in, double * out ) {

		double * data         = out;
		double * twiddle_data = twiddle;

        // depth 0 with vec reorder re-im-re-im-re-im-re-im   to   re-re-re-re-im-im-im-im
        {
            const size_t N8                           = size >> 3;

            for ( size_t n = 0; n < N8; n += vector_width ) {

                butterfly_zero( in, data, twiddle_data + n * 14, size );
                in   += vector_next;
                data += vector_next;
            }
            twiddle_data += 14 * N8; // 2 * 7 * N8
        }

		data = out;

		if ( depth > 3 ) fft_recursive( data, twiddle_data ); // unroll 512

		else {

			size_t step = size >> 3;
			const size_t top = depth - 1;

			  for ( size_t k = 1; k < top; ++k, step >>= 3 ) {

			      data = out;

			      const size_t N8 = step >> 3;

			      for ( size_t i = 0; i < size; i += step ) {

			          for ( size_t n = 0; n < N8; n += vector_width ) {

			              butterfly( data, twiddle_data + n * 14, step );
			              data += vector_next;
			          }
			          data += 14 * N8;
			      }
			      twiddle_data += 14 * N8;
			  }

				data = out;

		    for ( size_t n = 0; n < size; n += 32 ) {

		        butterfly_final( data );
		        data += 64;
		    }

		}

		revbin_permute( out, size );

    }

};

void print( double * a, double * b, int m ) {
    for ( int j = 0; j < m ; ++j ) {
        if (b[j] < 0)
            std::cout << std::fixed << std::setprecision(2) << a[j] << " - " << std::abs(b[j]) << "i    ";
        else
            std::cout << std::fixed << std::setprecision(2) << a[j] << " + " <<     b[j]  << "i    ";
    } std::cout << std::endl;
}

void test_output( size_t N ) {

	double * input = (double*)aligned_alloc(native_alignment, N * 2 * sizeof(double));
	double * output = (double*)aligned_alloc(native_alignment, N * 2 * sizeof(double));
	for ( uint32_t i = 0; i < N; ++i ) {
		input[i*2]   = (double)i;
		input[i*2+1] = (double)0.;
	}

	fft_plan FFT( N );
	FFT.fft ( input, output );
	FFT.fft_free();


	for ( uint32_t i = 0; i < 64; ++i ) {
		std::cout <<std::fixed<< std::setprecision(2) << output[i*2] << "   " << output[i*2+1] << "i  " ;
	}

	std::cout << std::endl;
}

uint32_t ipow( uint32_t base, uint32_t exp )
{
    uint32_t result = 1;
    for ( ; ; ) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if ( !exp )
            break;
        base *= base;
    }
    return result;
}

template <typename T>
constexpr inline std::uint32_t log_2( T n )
{
    return (n > 1) ? 1 + log_2(n >> 1) : 0;
}

template <typename T>
void time2n( uint32_t n, uint32_t base, uint32_t tests = 10 ) {

    uint32_t m = ipow(base,n); // !!! base must be same like radix number ( radix4 -> base 4 )

    using namespace std::chrono;

    std::vector<double> mflops(tests);
    std::vector<duration<double> > time(tests);

    T * input  = (T*)aligned_alloc(native_alignment, 2 * m * sizeof(T));
    T * output = (T*)aligned_alloc(native_alignment, 2 * m * sizeof(T));

	if ( base == 8 ) {

		fft_plan FFT( m );

		for ( uint32_t k = 0; k < tests; ++k ) {

		    for ( uint32_t i = 0; i < m; ++i ) {
                input[i*2]   = (T)0.;
                input[i*2+1] = (T)0.;
            }

		    high_resolution_clock::time_point t1 = high_resolution_clock::now();

		    FFT.fft ( input, output );

		    high_resolution_clock::time_point t2 = high_resolution_clock::now();

		    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

		    time[k] = time_span;

		    // 5 * log_2(n) / time

		    mflops[k] = 5 * m * log_2(m) / (std::log(2.0)*time_span.count() * 1000000.0F);

		}

        FFT.fft_free();
	}

    free(input);
    free(output);

    double mintime = time[0].count();
    double bestflops = mflops[0];

    for ( uint32_t k = 1; k < tests; ++k ) {
        if ( time[k].count() < mintime ) {
            mintime = time[k].count();
            bestflops = mflops[k];
        }
    }

    std::cout<<log_2(m)<<'\t'<<bestflops<<'\t'<<std::fixed <<std::setprecision(9)<<(mintime * 1000.0F)<<std::endl;

}

void test_speed ( void ) {
    std::cout<<"#size"<<'\t'<<"speed(mflops)"<<'\t'<<"time(ms) r4"<<std::endl;
    for ( int i = 2; i < 9; ++i ) time2n<double>(i,8);
}


int main(void)
{
		/*/
		test_output(64);
		test_output(512);
		test_output(4096);
		/*/
    test_speed();
		//*/
	return 0;
}
