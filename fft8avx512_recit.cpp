#include <bits/stdc++.h>
#include <immintrin.h>

#define CTRL 0b11011000
#define SHLO 0b01000100
#define SHHI 0b11101110

#define _MM512_FAKE_TRANSPOSE_8x8_PD(row0,row1,row2,row3,row4,row5,row6,row7) \
    do { \
        __m512d __t7, __t6, __t5, __t4, __t3, __t2, __t1, __t0; \
        __t0 = _mm512_unpacklo_pd((row0), (row1)); \
        __t1 = _mm512_unpackhi_pd((row0), (row1)); \
        __t2 = _mm512_unpacklo_pd((row2), (row3)); \
        __t3 = _mm512_unpackhi_pd((row2), (row3)); \
		__t4 = _mm512_unpacklo_pd((row4), (row5)); \
        __t5 = _mm512_unpackhi_pd((row4), (row5)); \
        __t6 = _mm512_unpacklo_pd((row6), (row7)); \
        __t7 = _mm512_unpackhi_pd((row6), (row7)); \
			(row0) = _mm512_permutex_pd(__t0, CTRL ); \
			(row1) = _mm512_permutex_pd(__t1, CTRL ); \
			(row2) = _mm512_permutex_pd(__t2, CTRL ); \
			(row3) = _mm512_permutex_pd(__t3, CTRL ); \
			(row4) = _mm512_permutex_pd(__t4, CTRL ); \
			(row5) = _mm512_permutex_pd(__t5, CTRL ); \
			(row6) = _mm512_permutex_pd(__t6, CTRL ); \
			(row7) = _mm512_permutex_pd(__t7, CTRL ); \
        __t0 = _mm512_unpacklo_pd((row0), (row2)); \
        __t2 = _mm512_unpackhi_pd((row0), (row2)); \
        __t1 = _mm512_unpacklo_pd((row1), (row3)); \
        __t3 = _mm512_unpackhi_pd((row1), (row3)); \
		__t4 = _mm512_unpacklo_pd((row4), (row6)); \
        __t6 = _mm512_unpackhi_pd((row4), (row6)); \
        __t5 = _mm512_unpacklo_pd((row5), (row7)); \
        __t7 = _mm512_unpackhi_pd((row5), (row7)); \
			(row0) = _mm512_shuffle_f64x2(__t0, __t4, SHLO ); \
			(row1) = _mm512_shuffle_f64x2(__t1, __t5, SHLO ); \
			(row2) = _mm512_shuffle_f64x2(__t0, __t4, SHHI ); \
			(row3) = _mm512_shuffle_f64x2(__t1, __t5, SHHI ); \
			(row4) = _mm512_shuffle_f64x2(__t2, __t6, SHLO ); \
			(row5) = _mm512_shuffle_f64x2(__t3, __t7, SHLO ); \
			(row6) = _mm512_shuffle_f64x2(__t2, __t6, SHHI ); \
			(row7) = _mm512_shuffle_f64x2(__t3, __t7, SHHI ); \
    } while (0)

#define _MM512_FAKE_TRANSPOSE_B_8x8_PD(row0,row1,row2,row3,row4,row5,row6,row7) \
    do { \
        __m512d __t7, __t6, __t5, __t4, __t3, __t2, __t1, __t0; \
            __t0 = _mm512_shuffle_f64x2((row0), (row4), SHLO ); \
			__t1 = _mm512_shuffle_f64x2((row1), (row5), SHLO ); \
			__t2 = _mm512_shuffle_f64x2((row2), (row6), SHLO ); \
			__t3 = _mm512_shuffle_f64x2((row3), (row7), SHLO ); \
			__t4 = _mm512_shuffle_f64x2((row0), (row4), SHHI ); \
			__t5 = _mm512_shuffle_f64x2((row1), (row5), SHHI ); \
			__t6 = _mm512_shuffle_f64x2((row2), (row6), SHHI ); \
			__t7 = _mm512_shuffle_f64x2((row3), (row7), SHHI ); \
        (row0) = _mm512_unpacklo_pd(__t0, __t1); \
        (row4) = _mm512_unpackhi_pd(__t0, __t1); \
        (row1) = _mm512_unpacklo_pd(__t2, __t3); \
        (row5) = _mm512_unpackhi_pd(__t2, __t3); \
		(row2) = _mm512_unpacklo_pd(__t4, __t5); \
        (row6) = _mm512_unpackhi_pd(__t4, __t5); \
        (row3) = _mm512_unpacklo_pd(__t6, __t7); \
        (row7) = _mm512_unpackhi_pd(__t6, __t7); \
            __t0 = _mm512_permutex_pd((row0), CTRL ); \
			__t1 = _mm512_permutex_pd((row1), CTRL ); \
			__t2 = _mm512_permutex_pd((row2), CTRL ); \
			__t3 = _mm512_permutex_pd((row3), CTRL ); \
			__t4 = _mm512_permutex_pd((row4), CTRL ); \
			__t5 = _mm512_permutex_pd((row5), CTRL ); \
			__t6 = _mm512_permutex_pd((row6), CTRL ); \
			__t7 = _mm512_permutex_pd((row7), CTRL ); \
        (row0) = _mm512_unpacklo_pd(__t0, __t1); \
        (row1) = _mm512_unpackhi_pd(__t0, __t1); \
        (row4) = _mm512_unpacklo_pd(__t2, __t3); \
        (row5) = _mm512_unpackhi_pd(__t2, __t3); \
		(row2) = _mm512_unpacklo_pd(__t4, __t5); \
        (row3) = _mm512_unpackhi_pd(__t4, __t5); \
        (row6) = _mm512_unpacklo_pd(__t6, __t7); \
        (row7) = _mm512_unpackhi_pd(__t6, __t7); \
    } while (0)

#define native_alignment 64
#define cache_fit 1024
#define vector_width 8
#define vector_next 16
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
        double rr , ii;

        sincos( 2 * M_PI * (stage_offs                   ) / size, &ii, &rr);
        twiddle[0]                =  rr;
        twiddle[vector_width]     = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs * 4 ) / size, &ii, &rr);
        twiddle[1]                =  rr;
        twiddle[1 + vector_width] = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs     ) / size, &ii, &rr);
        twiddle[2]                =  rr;
        twiddle[2 + vector_width] = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs * 5 ) / size, &ii, &rr);
        twiddle[3]                =  rr;
        twiddle[3 + vector_width] = -ii;

		sincos( 2 * M_PI * (stage_offs + blocks_offs * 2 ) / size, &ii, &rr);
        twiddle[4]                =  rr;
        twiddle[4 + vector_width]     = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs * 6 ) / size, &ii, &rr);
        twiddle[5]                =  rr;
        twiddle[5 + vector_width] = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs * 3 ) / size, &ii, &rr);
        twiddle[6]                =  rr;
        twiddle[6 + vector_width] = -ii;

        sincos( 2 * M_PI * (stage_offs + blocks_offs * 7 ) / size, &ii, &rr);
        twiddle[7]                =  rr;
        twiddle[7 + vector_width] = -ii;

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

        __m512d xmx = _mm512_load_pd( in 	          );
        __m512d xmy = _mm512_load_pd( in          + 8 );
        __m512d re0 = _mm512_unpacklo_pd( xmx, xmy );
        __m512d im0 = _mm512_unpackhi_pd( xmx, xmy );
                xmx = _mm512_load_pd( in + N8         );
                xmy = _mm512_load_pd( in + N8     + 8 );
        __m512d re1 = _mm512_unpacklo_pd( xmx, xmy );
        __m512d im1 = _mm512_unpackhi_pd( xmx, xmy );
                xmx = _mm512_load_pd( in + N8 * 2     );
                xmy = _mm512_load_pd( in + N8 * 2 + 8 );
        __m512d re2 = _mm512_unpacklo_pd( xmx, xmy );
        __m512d im2 = _mm512_unpackhi_pd( xmx, xmy );
                xmx = _mm512_load_pd( in + N8 * 3     );
                xmy = _mm512_load_pd( in + N8 * 3 + 8 );
        __m512d re3 = _mm512_unpacklo_pd( xmx, xmy );
        __m512d im3 = _mm512_unpackhi_pd( xmx, xmy );
                xmx = _mm512_load_pd( in + N8 * 4     );
                xmy = _mm512_load_pd( in + N8 * 4 + 8 );
        __m512d re4 = _mm512_unpacklo_pd( xmx, xmy );
        __m512d im4 = _mm512_unpackhi_pd( xmx, xmy );
                xmx = _mm512_load_pd( in + N8 * 5     );
                xmy = _mm512_load_pd( in + N8 * 5 + 8 );
        __m512d re5 = _mm512_unpacklo_pd( xmx, xmy );
        __m512d im5 = _mm512_unpackhi_pd( xmx, xmy );
                xmx = _mm512_load_pd( in + N8 * 6     );
                xmy = _mm512_load_pd( in + N8 * 6 + 8 );
        __m512d re6 = _mm512_unpacklo_pd( xmx, xmy );
        __m512d im6 = _mm512_unpackhi_pd( xmx, xmy );
                xmx = _mm512_load_pd( in + N8 * 7     );
                xmy = _mm512_load_pd( in + N8 * 7 + 8 );
        __m512d re7 = _mm512_unpacklo_pd( xmx, xmy );
        __m512d im7 = _mm512_unpackhi_pd( xmx, xmy );

        /*_mm512_store_pd( data     ,         re0);
        _mm512_store_pd( data + 8 ,         im0);
        _mm512_store_pd( data + N8,         re1);
        _mm512_store_pd( data + N8 + 8,     im1);
        _mm512_store_pd( data + 2*N8,       re2);
        _mm512_store_pd( data + 2*N8 + 8,   im2);
        _mm512_store_pd( data + 3*N8,       re3);
        _mm512_store_pd( data + 3*N8 + 8,   im3);
        _mm512_store_pd( data + 4*N8,       re4);
        _mm512_store_pd( data + 4*N8 + 8,   im4);
        _mm512_store_pd( data + 5*N8,       re5);
        _mm512_store_pd( data + 5*N8 + 8,   im5);
        _mm512_store_pd( data + 6*N8,       re6);
        _mm512_store_pd( data + 6*N8 + 8,   im6);
        _mm512_store_pd( data + 7*N8,       re7);
        _mm512_store_pd( data + 7*N8 + 8,   im7);*/

        __m512d sum04re = _mm512_add_pd( re0, re4 );
        __m512d sum04im = _mm512_add_pd( im0, im4 );
        __m512d sum15re = _mm512_add_pd( re1, re5 );
        __m512d sum15im = _mm512_add_pd( im1, im5 );
        __m512d sum26re = _mm512_add_pd( re2, re6 );
        __m512d sum26im = _mm512_add_pd( im2, im6 );
        __m512d sum37re = _mm512_add_pd( re3, re7 );
        __m512d sum37im = _mm512_add_pd( im3, im7 );

        __m512d xma, xmb, xE4;

        //0,4
        xma = _mm512_add_pd(sum04re, sum26re);
        xmb = _mm512_add_pd(sum37re, sum15re);

        _mm512_store_pd( data, _mm512_add_pd( xma, xmb ) );
        __m512d w4r = _mm512_sub_pd(xma, xmb);

        xma = _mm512_add_pd(sum04im, sum26im);
        xmb = _mm512_add_pd(sum37im, sum15im);

        _mm512_store_pd( data + 8, _mm512_add_pd( xma, xmb ) );
        __m512d w4i = _mm512_sub_pd( xma, xmb );

        __m512d tw4r = _mm512_load_pd(twiddle + 6 * vector_width );
        __m512d tw4i = _mm512_load_pd(twiddle + 7 * vector_width );

        __m512d tw4or = _mm512_sub_pd( _mm512_mul_pd(w4r, tw4r), _mm512_mul_pd(w4i, tw4i) );
        __m512d tw4oi = _mm512_add_pd( _mm512_mul_pd(w4i, tw4r), _mm512_mul_pd(w4r, tw4i) );

        _mm512_store_pd( data + N8    , tw4or );
        _mm512_store_pd( data + N8 + 8, tw4oi );

        //2,6
        xma = _mm512_sub_pd(sum04re, sum26re);
        xmb = _mm512_sub_pd(sum37im, sum15im);
        __m512d w2r = _mm512_sub_pd(xma, xmb);
        __m512d w6r = _mm512_add_pd(xma, xmb);
        xma = _mm512_sub_pd(sum04im, sum26im);
        xmb = _mm512_sub_pd(sum37re, sum15re);
        __m512d w2i = _mm512_add_pd(xma, xmb);
        __m512d w6i = _mm512_sub_pd(xma, xmb);


        __m512d tw2r = _mm512_load_pd(twiddle + 2  * vector_width );
        __m512d tw2i = _mm512_load_pd(twiddle + 3  * vector_width );
        __m512d tw6r = _mm512_load_pd(twiddle + 10 * vector_width );
        __m512d tw6i = _mm512_load_pd(twiddle + 11 * vector_width );

         __m512d tw2or = _mm512_sub_pd( _mm512_mul_pd(w2r, tw2r), _mm512_mul_pd(w2i, tw2i) );
         __m512d tw2oi = _mm512_add_pd( _mm512_mul_pd(w2i, tw2r), _mm512_mul_pd(w2r, tw2i) );
         __m512d tw6or = _mm512_sub_pd( _mm512_mul_pd(w6r, tw6r), _mm512_mul_pd(w6i, tw6i) );
         __m512d tw6oi = _mm512_add_pd( _mm512_mul_pd(w6i, tw6r), _mm512_mul_pd(w6r, tw6i) );

        _mm512_store_pd( data + N8 * 2    , tw2or );
        _mm512_store_pd( data + N8 * 2 + 8, tw2oi );
        _mm512_store_pd( data + N8 * 3    , tw6or );
        _mm512_store_pd( data + N8 * 3 + 8, tw6oi );

        __m512d diff04re = _mm512_sub_pd( re0, re4 );
        __m512d diff04im = _mm512_sub_pd( im0, im4 );
        __m512d diff15re = _mm512_sub_pd( re1, re5 );
        __m512d diff15im = _mm512_sub_pd( im1, im5 );
        __m512d diff26re = _mm512_sub_pd( re2, re6 );
        __m512d diff26im = _mm512_sub_pd( im2, im6 );
        __m512d diff37re = _mm512_sub_pd( re3, re7 );
        __m512d diff37im = _mm512_sub_pd( im3, im7 );

        //3,7
        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_add_pd(_mm512_add_pd(diff37re, diff37im), _mm512_sub_pd(diff15im, diff15re)) );

        xma = _mm512_sub_pd(diff04re, diff26im);

        __m512d w3r = _mm512_add_pd(xma, xE4);
        __m512d w7r = _mm512_sub_pd(xma, xE4);

        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_sub_pd(_mm512_sub_pd(diff37im, diff37re), _mm512_add_pd(diff15re, diff15im)) );

        xma = _mm512_add_pd(diff04im, diff26re);

        __m512d w3i = _mm512_add_pd(xma, xE4);
        __m512d w7i = _mm512_sub_pd(xma, xE4);

        __m512d tw3r = _mm512_load_pd(twiddle + 4  * vector_width );
        __m512d tw3i = _mm512_load_pd(twiddle + 5  * vector_width );
        __m512d tw7r = _mm512_load_pd(twiddle + 12 * vector_width );
        __m512d tw7i = _mm512_load_pd(twiddle + 13 * vector_width );

        __m512d tw3or = _mm512_sub_pd( _mm512_mul_pd(w3r, tw3r), _mm512_mul_pd(w3i, tw3i) );
        __m512d tw3oi = _mm512_add_pd( _mm512_mul_pd(w3i, tw3r), _mm512_mul_pd(w3r, tw3i) );
        __m512d tw7or = _mm512_sub_pd( _mm512_mul_pd(w7r, tw7r), _mm512_mul_pd(w7i, tw7i) );
        __m512d tw7oi = _mm512_add_pd( _mm512_mul_pd(w7i, tw7r), _mm512_mul_pd(w7r, tw7i) );

        _mm512_store_pd( data + N8 * 6    , tw3or );
        _mm512_store_pd( data + N8 * 6 + 8, tw3oi );
        _mm512_store_pd( data + N8 * 7    , tw7or );
        _mm512_store_pd( data + N8 * 7 + 8, tw7oi );

        //1,5
        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_add_pd(_mm512_sub_pd(diff37im, diff37re), _mm512_add_pd(diff15re, diff15im)) );

        xma = _mm512_add_pd(diff04re, diff26im);

        __m512d w5r = _mm512_sub_pd(xma, xE4);
        __m512d w1r = _mm512_add_pd(xma, xE4);

        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_sub_pd(_mm512_add_pd(diff37re, diff37im), _mm512_sub_pd(diff15im, diff15re)) );

        xma = _mm512_sub_pd(diff04im, diff26re);

        __m512d w5i = _mm512_add_pd(xma, xE4);
        __m512d w1i = _mm512_sub_pd(xma, xE4);

        __m512d tw1r = _mm512_load_pd(twiddle            );
        __m512d tw1i = _mm512_load_pd(twiddle + vector_width    );
        __m512d tw5r = _mm512_load_pd(twiddle + 8 * vector_width);
        __m512d tw5i = _mm512_load_pd(twiddle + 9 * vector_width);

        __m512d tw1or = _mm512_sub_pd( _mm512_mul_pd(w1r, tw1r), _mm512_mul_pd(w1i, tw1i) );
        __m512d tw1oi = _mm512_add_pd( _mm512_mul_pd(w1i, tw1r), _mm512_mul_pd(w1r, tw1i) );
        __m512d tw5or = _mm512_sub_pd( _mm512_mul_pd(w5r, tw5r), _mm512_mul_pd(w5i, tw5i) );
        __m512d tw5oi = _mm512_add_pd( _mm512_mul_pd(w5i, tw5r), _mm512_mul_pd(w5r, tw5i) );

        _mm512_store_pd( data + N8 * 4    , tw1or );
        _mm512_store_pd( data + N8 * 4 + 8, tw1oi );
        _mm512_store_pd( data + N8 * 5    , tw5or );
        _mm512_store_pd( data + N8 * 5 + 8, tw5oi );
    }

    inline void butterfly( double * data, const double * twiddle, uint32_t stage_size ) {

        const size_t N8 = stage_size >> 2;

        __m512d re0 = _mm512_load_pd(data              );
        __m512d im0 = _mm512_load_pd(data          + 8 );
        __m512d re1 = _mm512_load_pd(data + N8         );
        __m512d im1 = _mm512_load_pd(data + N8     + 8 );
        __m512d re2 = _mm512_load_pd(data + N8 * 2     );
        __m512d im2 = _mm512_load_pd(data + N8 * 2 + 8 );
        __m512d re3 = _mm512_load_pd(data + N8 * 3     );
        __m512d im3 = _mm512_load_pd(data + N8 * 3 + 8 );
        __m512d re4 = _mm512_load_pd(data + N8 * 4     );
        __m512d im4 = _mm512_load_pd(data + N8 * 4 + 8 );
        __m512d re5 = _mm512_load_pd(data + N8 * 5     );
        __m512d im5 = _mm512_load_pd(data + N8 * 5 + 8 );
        __m512d re6 = _mm512_load_pd(data + N8 * 6     );
        __m512d im6 = _mm512_load_pd(data + N8 * 6 + 8 );
        __m512d re7 = _mm512_load_pd(data + N8 * 7     );
        __m512d im7 = _mm512_load_pd(data + N8 * 7 + 8 );

        __m512d sum04re = _mm512_add_pd( re0, re4 );
        __m512d sum04im = _mm512_add_pd( im0, im4 );
        __m512d sum15re = _mm512_add_pd( re1, re5 );
        __m512d sum15im = _mm512_add_pd( im1, im5 );
        __m512d sum26re = _mm512_add_pd( re2, re6 );
        __m512d sum26im = _mm512_add_pd( im2, im6 );
        __m512d sum37re = _mm512_add_pd( re3, re7 );
        __m512d sum37im = _mm512_add_pd( im3, im7 );

        __m512d xma, xmb, xE4;

        //0,4
        xma = _mm512_add_pd(sum04re, sum26re);
        xmb = _mm512_add_pd(sum37re, sum15re);

        _mm512_store_pd( data, _mm512_add_pd( xma, xmb ) );
        __m512d w4r = _mm512_sub_pd(xma, xmb);

        xma = _mm512_add_pd(sum04im, sum26im);
        xmb = _mm512_add_pd(sum37im, sum15im);

        _mm512_store_pd( data + 8, _mm512_add_pd( xma, xmb ) );
        __m512d w4i = _mm512_sub_pd( xma, xmb );

        __m512d tw4r = _mm512_load_pd(twiddle + 6 * vector_width );
        __m512d tw4i = _mm512_load_pd(twiddle + 7 * vector_width );

        __m512d tw4or = _mm512_sub_pd( _mm512_mul_pd(w4r, tw4r), _mm512_mul_pd(w4i, tw4i) );
        __m512d tw4oi = _mm512_add_pd( _mm512_mul_pd(w4i, tw4r), _mm512_mul_pd(w4r, tw4i) );

        _mm512_store_pd( data + N8    , tw4or );
        _mm512_store_pd( data + N8 + 8, tw4oi );

        //2,6
        xma = _mm512_sub_pd(sum04re, sum26re);
        xmb = _mm512_sub_pd(sum37im, sum15im);
        __m512d w2r = _mm512_sub_pd(xma, xmb);
        __m512d w6r = _mm512_add_pd(xma, xmb);
        xma = _mm512_sub_pd(sum04im, sum26im);
        xmb = _mm512_sub_pd(sum37re, sum15re);
        __m512d w2i = _mm512_add_pd(xma, xmb);
        __m512d w6i = _mm512_sub_pd(xma, xmb);


        __m512d tw2r = _mm512_load_pd(twiddle + 2  * vector_width );
        __m512d tw2i = _mm512_load_pd(twiddle + 3  * vector_width );
        __m512d tw6r = _mm512_load_pd(twiddle + 10 * vector_width );
        __m512d tw6i = _mm512_load_pd(twiddle + 11 * vector_width );

         __m512d tw2or = _mm512_sub_pd( _mm512_mul_pd(w2r, tw2r), _mm512_mul_pd(w2i, tw2i) );
         __m512d tw2oi = _mm512_add_pd( _mm512_mul_pd(w2i, tw2r), _mm512_mul_pd(w2r, tw2i) );
         __m512d tw6or = _mm512_sub_pd( _mm512_mul_pd(w6r, tw6r), _mm512_mul_pd(w6i, tw6i) );
         __m512d tw6oi = _mm512_add_pd( _mm512_mul_pd(w6i, tw6r), _mm512_mul_pd(w6r, tw6i) );

        _mm512_store_pd( data + N8 * 2    , tw2or );
        _mm512_store_pd( data + N8 * 2 + 8, tw2oi );
        _mm512_store_pd( data + N8 * 3    , tw6or );
        _mm512_store_pd( data + N8 * 3 + 8, tw6oi );

        __m512d diff04re = _mm512_sub_pd( re0, re4 );
        __m512d diff04im = _mm512_sub_pd( im0, im4 );
        __m512d diff15re = _mm512_sub_pd( re1, re5 );
        __m512d diff15im = _mm512_sub_pd( im1, im5 );
        __m512d diff26re = _mm512_sub_pd( re2, re6 );
        __m512d diff26im = _mm512_sub_pd( im2, im6 );
        __m512d diff37re = _mm512_sub_pd( re3, re7 );
        __m512d diff37im = _mm512_sub_pd( im3, im7 );

        //3,7
        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_add_pd(_mm512_add_pd(diff37re, diff37im), _mm512_sub_pd(diff15im, diff15re)) );

        xma = _mm512_sub_pd(diff04re, diff26im);

        __m512d w3r = _mm512_add_pd(xma, xE4);
        __m512d w7r = _mm512_sub_pd(xma, xE4);

        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_sub_pd(_mm512_sub_pd(diff37im, diff37re), _mm512_add_pd(diff15re, diff15im)) );

        xma = _mm512_add_pd(diff04im, diff26re);

        __m512d w3i = _mm512_add_pd(xma, xE4);
        __m512d w7i = _mm512_sub_pd(xma, xE4);

        __m512d tw3r = _mm512_load_pd(twiddle + 4  * vector_width );
        __m512d tw3i = _mm512_load_pd(twiddle + 5  * vector_width );
        __m512d tw7r = _mm512_load_pd(twiddle + 12 * vector_width );
        __m512d tw7i = _mm512_load_pd(twiddle + 13 * vector_width );

        __m512d tw3or = _mm512_sub_pd( _mm512_mul_pd(w3r, tw3r), _mm512_mul_pd(w3i, tw3i) );
        __m512d tw3oi = _mm512_add_pd( _mm512_mul_pd(w3i, tw3r), _mm512_mul_pd(w3r, tw3i) );
        __m512d tw7or = _mm512_sub_pd( _mm512_mul_pd(w7r, tw7r), _mm512_mul_pd(w7i, tw7i) );
        __m512d tw7oi = _mm512_add_pd( _mm512_mul_pd(w7i, tw7r), _mm512_mul_pd(w7r, tw7i) );

        _mm512_store_pd( data + N8 * 6    , tw3or );
        _mm512_store_pd( data + N8 * 6 + 8, tw3oi );
        _mm512_store_pd( data + N8 * 7    , tw7or );
        _mm512_store_pd( data + N8 * 7 + 8, tw7oi );

        //1,5
        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_add_pd(_mm512_sub_pd(diff37im, diff37re), _mm512_add_pd(diff15re, diff15im)) );

        xma = _mm512_add_pd(diff04re, diff26im);

        __m512d w5r = _mm512_sub_pd(xma, xE4);
        __m512d w1r = _mm512_add_pd(xma, xE4);

        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_sub_pd(_mm512_add_pd(diff37re, diff37im), _mm512_sub_pd(diff15im, diff15re)) );

        xma = _mm512_sub_pd(diff04im, diff26re);

        __m512d w5i = _mm512_add_pd(xma, xE4);
        __m512d w1i = _mm512_sub_pd(xma, xE4);

        __m512d tw1r = _mm512_load_pd(twiddle            );
        __m512d tw1i = _mm512_load_pd(twiddle + vector_width    );
        __m512d tw5r = _mm512_load_pd(twiddle + 8 * vector_width);
        __m512d tw5i = _mm512_load_pd(twiddle + 9 * vector_width);

        __m512d tw1or = _mm512_sub_pd( _mm512_mul_pd(w1r, tw1r), _mm512_mul_pd(w1i, tw1i) );
        __m512d tw1oi = _mm512_add_pd( _mm512_mul_pd(w1i, tw1r), _mm512_mul_pd(w1r, tw1i) );
        __m512d tw5or = _mm512_sub_pd( _mm512_mul_pd(w5r, tw5r), _mm512_mul_pd(w5i, tw5i) );
        __m512d tw5oi = _mm512_add_pd( _mm512_mul_pd(w5i, tw5r), _mm512_mul_pd(w5r, tw5i) );

        _mm512_store_pd( data + N8 * 4    , tw1or );
        _mm512_store_pd( data + N8 * 4 + 8, tw1oi );
        _mm512_store_pd( data + N8 * 5    , tw5or );
        _mm512_store_pd( data + N8 * 5 + 8, tw5oi );

    }

    inline void butterfly_final( double * data ) {

        __m512d re0 = _mm512_load_pd(data     );
        __m512d im0 = _mm512_load_pd(data + 8 );
        __m512d re1 = _mm512_load_pd(data + 16);
        __m512d im1 = _mm512_load_pd(data + 24);
        __m512d re2 = _mm512_load_pd(data + 32);
        __m512d im2 = _mm512_load_pd(data + 40);
        __m512d re3 = _mm512_load_pd(data + 48);
        __m512d im3 = _mm512_load_pd(data + 56);
        __m512d re4 = _mm512_load_pd(data + 64);
        __m512d im4 = _mm512_load_pd(data + 72);
        __m512d re5 = _mm512_load_pd(data + 80);
        __m512d im5 = _mm512_load_pd(data + 88);
        __m512d re6 = _mm512_load_pd(data + 96);
        __m512d im6 = _mm512_load_pd(data + 104);
        __m512d re7 = _mm512_load_pd(data + 112);
        __m512d im7 = _mm512_load_pd(data + 120);

        _MM512_FAKE_TRANSPOSE_8x8_PD(re0, re1, re2, re3, re4, re5, re6, re7);
        _MM512_FAKE_TRANSPOSE_8x8_PD(im0, im1, im2, im3, im4, im5, im6, im7);

        __m512d sum04re = _mm512_add_pd( re0, re1 );
        __m512d sum04im = _mm512_add_pd( im0, im1 );
        __m512d sum15re = _mm512_add_pd( re4, re5 );
        __m512d sum15im = _mm512_add_pd( im4, im5 );
        __m512d sum26re = _mm512_add_pd( re2, re3 );
        __m512d sum26im = _mm512_add_pd( im2, im3 );
        __m512d sum37re = _mm512_add_pd( re6, re7 );
        __m512d sum37im = _mm512_add_pd( im6, im7 );

        __m512d xma, xmb, xE4;

        //0,4
        xma = _mm512_add_pd(sum04re, sum26re);
        xmb = _mm512_add_pd(sum37re, sum15re);

        __m512d w0r = _mm512_add_pd( xma, xmb );
        __m512d w4r = _mm512_sub_pd( xma, xmb );

        xma = _mm512_add_pd(sum04im, sum26im);
        xmb = _mm512_add_pd(sum37im, sum15im);

        __m512d w0i = _mm512_add_pd( xma, xmb );
        __m512d w4i = _mm512_sub_pd( xma, xmb );

        //2,6
        xma = _mm512_sub_pd(sum04re, sum26re);
        xmb = _mm512_sub_pd(sum37im, sum15im);
        __m512d w2r = _mm512_sub_pd(xma, xmb);
        __m512d w6r = _mm512_add_pd(xma, xmb);
        xma = _mm512_sub_pd(sum04im, sum26im);
        xmb = _mm512_sub_pd(sum37re, sum15re);
        __m512d w2i = _mm512_add_pd(xma, xmb);
        __m512d w6i = _mm512_sub_pd(xma, xmb);

        __m512d diff04re = _mm512_sub_pd( re0, re1 );
        __m512d diff04im = _mm512_sub_pd( im0, im1 );
        __m512d diff15re = _mm512_sub_pd( re4, re5 );
        __m512d diff15im = _mm512_sub_pd( im4, im5 );
        __m512d diff26re = _mm512_sub_pd( re2, re3 );
        __m512d diff26im = _mm512_sub_pd( im2, im3 );
        __m512d diff37re = _mm512_sub_pd( re6, re7 );
        __m512d diff37im = _mm512_sub_pd( im6, im7 );

        //3,7
        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_add_pd(_mm512_add_pd(diff37re, diff37im), _mm512_sub_pd(diff15im, diff15re)) );

        xma = _mm512_sub_pd(diff04re, diff26im);

        __m512d w3r = _mm512_add_pd(xma, xE4);
        __m512d w7r = _mm512_sub_pd(xma, xE4);

        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_sub_pd(_mm512_sub_pd(diff37im, diff37re), _mm512_add_pd(diff15re, diff15im)) );

        xma = _mm512_add_pd(diff04im, diff26re);

        __m512d w3i = _mm512_add_pd(xma, xE4);
        __m512d w7i = _mm512_sub_pd(xma, xE4);

        //1,5
        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_add_pd(_mm512_sub_pd(diff37im, diff37re), _mm512_add_pd(diff15re, diff15im)) );

        xma = _mm512_add_pd(diff04re, diff26im);

        __m512d w5r = _mm512_sub_pd(xma, xE4);
        __m512d w1r = _mm512_add_pd(xma, xE4);

        xE4 = _mm512_mul_pd( _mm512_set1_pd(E4), _mm512_sub_pd(_mm512_add_pd(diff37re, diff37im), _mm512_sub_pd(diff15im, diff15re)) );

        xma = _mm512_sub_pd(diff04im, diff26re);

        __m512d w5i = _mm512_add_pd(xma, xE4);
        __m512d w1i = _mm512_sub_pd(xma, xE4);

        _MM512_FAKE_TRANSPOSE_B_8x8_PD(w0r, w4r, w1r, w5r, w2r, w6r, w3r, w7r);
        _MM512_FAKE_TRANSPOSE_B_8x8_PD(w0i, w4i, w1i, w5i, w2i, w6i, w3i, w7i);

        _mm512_store_pd( data     , _mm512_unpacklo_pd(w0r,w0i));
        _mm512_store_pd( data + 8 , _mm512_unpackhi_pd(w0r,w0i));
        _mm512_store_pd( data + 16, _mm512_unpacklo_pd(w4r,w4i));
        _mm512_store_pd( data + 24, _mm512_unpackhi_pd(w4r,w4i));
        _mm512_store_pd( data + 32, _mm512_unpacklo_pd(w1r,w1i));
        _mm512_store_pd( data + 40, _mm512_unpackhi_pd(w1r,w1i));
        _mm512_store_pd( data + 48, _mm512_unpacklo_pd(w5r,w5i));
        _mm512_store_pd( data + 56, _mm512_unpackhi_pd(w5r,w5i));
        _mm512_store_pd( data + 64, _mm512_unpacklo_pd(w2r,w2i));
        _mm512_store_pd( data + 72, _mm512_unpackhi_pd(w2r,w2i));
        _mm512_store_pd( data + 80, _mm512_unpacklo_pd(w6r,w6i));
        _mm512_store_pd( data + 88, _mm512_unpackhi_pd(w6r,w6i));
        _mm512_store_pd( data + 96, _mm512_unpacklo_pd(w3r,w3i));
        _mm512_store_pd( data + 104, _mm512_unpackhi_pd(w3r,w3i));
        _mm512_store_pd( data + 112, _mm512_unpacklo_pd(w7r,w7i));
        _mm512_store_pd( data + 120, _mm512_unpackhi_pd(w7r,w7i));

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

					for ( size_t n = 0; n < 512; n += 64 ) {

						butterfly_final( subdata );
						subdata += 128;
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

		    for ( size_t n = 0; n < size; n += 64 ) {

		        butterfly_final( data );
		        data += 128;
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


	for ( uint32_t i = 0; i < N; ++i ) {
		std::cout <<std::fixed<< std::setprecision(2) << output[i*2] << "   " << output[i*2+1] << "i  " ;
	}

	std::cout << std::endl;
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
		//test_output(64);
		//test_output(512);
		test_output(4096);
		/*/
    test_speed();
		//*/
	return 0;
}
