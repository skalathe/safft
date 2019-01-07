#include <bits/stdc++.h>
#include <immintrin.h>

#define CTRL 0b11011000
#define SHLO 0b01000100
#define SHHI 0b11101110

#define _MM512_FAKE_TRANSPOSE_4x8_PD(row0,row1,row2,row3) \
    do { \
        __m512d __t3, __t2, __t1, __t0; \
        __t0 = _mm512_shuffle_f64x2((row0), (row1), SHLO ); \
        __t1 = _mm512_shuffle_f64x2((row2), (row3), SHLO ); \
        __t2 = _mm512_shuffle_f64x2((row0), (row1), SHHI ); \
        __t3 = _mm512_shuffle_f64x2((row2), (row3), SHHI ); \
			__t3 = _mm512_permutex_pd(__t3, CTRL ); \
			__t2 = _mm512_permutex_pd(__t2, CTRL ); \
			__t1 = _mm512_permutex_pd(__t1, CTRL ); \
			__t0 = _mm512_permutex_pd(__t0, CTRL ); \
        (row0) = _mm512_unpacklo_pd(__t0, __t1); \
        (row1) = _mm512_unpackhi_pd(__t0, __t1); \
        (row2) = _mm512_unpacklo_pd(__t2, __t3); \
        (row3) = _mm512_unpackhi_pd(__t2, __t3); \
    } while (0)

#define _MM512_FAKE_TRANSPOSE_B_4x8_PD(row0,row1,row2,row3) \
    do { \
        __m512d __t3, __t2, __t1, __t0; \
        __t0 = _mm512_shuffle_f64x2((row0), (row1), SHLO ); \
        __t1 = _mm512_shuffle_f64x2((row2), (row3), SHLO ); \
        __t2 = _mm512_shuffle_f64x2((row0), (row1), SHHI ); \
        __t3 = _mm512_shuffle_f64x2((row2), (row3), SHHI ); \
            (row0) = _mm512_unpacklo_pd(__t0, __t1); \
            (row2) = _mm512_unpackhi_pd(__t0, __t1); \
            (row1) = _mm512_unpacklo_pd(__t2, __t3); \
            (row3) = _mm512_unpackhi_pd(__t2, __t3); \
        (row3) = _mm512_permutex_pd((row3), CTRL ); \
        (row2) = _mm512_permutex_pd((row2), CTRL ); \
        (row1) = _mm512_permutex_pd((row1), CTRL ); \
        (row0) = _mm512_permutex_pd((row0), CTRL ); \
    } while (0)

#define native_alignment 64
#define cache_fit 1024
#define vector_width 8
#define vector_next 16
#define fft_MAX 32

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

    inline void initialize_twiddles( double *& twiddle, size_t stage_offs, size_t stage_offs_, size_t blocks_offs, size_t size )
    {
        double rr, ii ;

        sincos( 2 * M_PI * (stage_offs                    ) / size, &ii, &rr);
        twiddle[0]                =  rr;
        twiddle[vector_width]     = -ii;
        sincos( 2 * M_PI * (stage_offs_                   ) / size, &ii, &rr);
        twiddle[1]                =  rr;
        twiddle[1 + vector_width] = -ii;

        sincos( 2 * M_PI * (stage_offs  + blocks_offs     ) / size, &ii, &rr);
        twiddle[2]                =  rr;
        twiddle[2 + vector_width] = -ii;
        sincos( 2 * M_PI * (stage_offs_ + blocks_offs     ) / size, &ii, &rr);
        twiddle[3]                =  rr;
        twiddle[3 + vector_width] = -ii;

		sincos( 2 * M_PI * (stage_offs  + blocks_offs * 2 ) / size, &ii, &rr);
        twiddle[4]                =  rr;
        twiddle[4 + vector_width] = -ii;
        sincos( 2 * M_PI * (stage_offs_ + blocks_offs * 2 ) / size, &ii, &rr);
        twiddle[5]                =  rr;
        twiddle[5 + vector_width] = -ii;

        sincos( 2 * M_PI * (stage_offs  + blocks_offs * 3 ) / size, &ii, &rr);
        twiddle[6]                =  rr;
        twiddle[6 + vector_width] = -ii;
        sincos( 2 * M_PI * (stage_offs_ + blocks_offs * 3 ) / size, &ii, &rr);
        twiddle[7]                =  rr;
        twiddle[7 + vector_width] = -ii;

        twiddle += 2 * vector_width;
    }

    fft_plan ( size_t size ) : size ( size ) {

				size_t twiddle_size = 0;

				for ( size_t s = size; s > 4; s /= 4 ) twiddle_size |= s;

        twiddle = (double*) aligned_alloc( native_alignment, (2 * twiddle_size + 4) * sizeof(double) );

        double * twiddleinit = twiddle;

        depth = 2;
        size_t blocks = 1;
        for ( size_t stage_size = size; stage_size > 16; stage_size >>= 2, blocks <<= 2 ) {

        	depth++;

            for ( size_t n = 0; n < stage_size >> 2; n += vector_width ) {

                initialize_twiddles( twiddleinit, n * blocks    , (n+4) * blocks    , blocks    , size );
                initialize_twiddles( twiddleinit, n * blocks * 2, (n+4) * blocks * 2, blocks * 2, size );
                initialize_twiddles( twiddleinit, n * blocks * 3, (n+4) * blocks * 3, blocks * 3, size );
            }
        }

        for ( size_t n = 0; n < 4; n += vector_width ) {

            initialize_twiddles( twiddleinit, n * blocks    , n * blocks    , blocks    , size );
            initialize_twiddles( twiddleinit, n * blocks * 2, n * blocks * 2, blocks * 2, size );
            initialize_twiddles( twiddleinit, n * blocks * 3, n * blocks * 3, blocks * 3, size );
        }
    }

    void fft_free ( void ) {

        free(twiddle);

    }

    inline void butterfly_zero( const double * in, double * out, const double * twiddle, uint32_t stage_size ) {

        const size_t N4 = stage_size >> 1;

        // 0 ..N4| 1 ..2N4| 2 ..3N4| 3

        __m512d xmx = _mm512_load_pd( in 	          );
        __m512d xmy = _mm512_load_pd( in          + 8 );
        __m512d xm4 = _mm512_unpacklo_pd( xmx, xmy );       // 0 re
        __m512d xm5 = _mm512_unpackhi_pd( xmx, xmy );       // 0 im

                xmx = _mm512_load_pd( in + N4 * 2     );
                xmy = _mm512_load_pd( in + N4 * 2 + 8 );

        __m512d xmz = _mm512_unpacklo_pd( xmx, xmy );

        __m512d xm0 = _mm512_add_pd( xm4, xmz );     // +02 re
                xm4 = _mm512_sub_pd( xm4, xmz );     // -02 re

                xmz = _mm512_unpackhi_pd( xmx, xmy );
        __m512d xm1 = _mm512_add_pd( xm5, xmz );     // +02 im
                xm5 = _mm512_sub_pd( xm5, xmz );     // -02 im

                xmx = _mm512_load_pd( in + N4         );
                xmy = _mm512_load_pd( in + N4     + 8 );
        __m512d xm6 = _mm512_unpacklo_pd( xmx, xmy );       // 1 re
        __m512d xm7 = _mm512_unpackhi_pd( xmx, xmy );       // 1 im


                xmx = _mm512_load_pd( in + N4 * 3     );
                xmy = _mm512_load_pd( in + N4 * 3 + 8 );

                xmz = _mm512_unpacklo_pd( xmx, xmy );
        __m512d xm2 = _mm512_add_pd( xm6, xmz );     // +13 re
                xm6 = _mm512_sub_pd( xm6, xmz );     // -13 re

                xmz = _mm512_unpackhi_pd( xmx, xmy );
        __m512d xm3 = _mm512_add_pd( xm7, xmz );     // +13 im
                xm7 = _mm512_add_pd( xm7, xmz );     // -13 im


        __m512d w2r = _mm512_sub_pd( xm0, xm2 );
        __m512d w2i = _mm512_sub_pd( xm1, xm3 );

        __m512d tw2r = _mm512_load_pd( twiddle + vector_width * 2 );
        __m512d tw2i = _mm512_load_pd( twiddle + vector_width * 3 );

        __m512d tw2or = _mm512_sub_pd( _mm512_mul_pd(w2r, tw2r), _mm512_mul_pd(w2i, tw2i));
        __m512d tw2oi = _mm512_add_pd( _mm512_mul_pd(w2i, tw2r), _mm512_mul_pd(w2r, tw2i));

        __m512d tw1r = _mm512_load_pd(twiddle                    );
        __m512d tw1i = _mm512_load_pd(twiddle + vector_width     );

        __m512d tw3r = _mm512_load_pd(twiddle + vector_width * 4 );
        __m512d tw3i = _mm512_load_pd(twiddle + vector_width * 5 );

        __m512d w3r = _mm512_sub_pd( xm4, xm7 );
        __m512d w3i = _mm512_add_pd( xm5, xm6 );
        __m512d w1r = _mm512_add_pd( xm4, xm7 );
        __m512d w1i = _mm512_sub_pd( xm5, xm6 );

        __m512d tw1or = _mm512_sub_pd( _mm512_mul_pd(w1r, tw1r), _mm512_mul_pd(w1i, tw1i));
        __m512d tw1oi = _mm512_add_pd( _mm512_mul_pd(w1i, tw1r), _mm512_mul_pd(w1r, tw1i));

        __m512d tw3or = _mm512_sub_pd( _mm512_mul_pd(w3r, tw3r), _mm512_mul_pd(w3i, tw3i));
        __m512d tw3oi = _mm512_add_pd( _mm512_mul_pd(w3i, tw3r), _mm512_mul_pd(w3r, tw3i));

        _mm512_store_pd( out             , _mm512_add_pd( xm0, xm2 ));
        _mm512_store_pd( out          + 8, _mm512_add_pd( xm1, xm3 ));

        _mm512_store_pd( out + N4        , tw2or );
        _mm512_store_pd( out + N4     + 8, tw2oi );

        _mm512_store_pd( out + N4 * 2    , tw1or );
        _mm512_store_pd( out + N4 * 2 + 8, tw1oi );

        _mm512_store_pd( out + N4 * 3    , tw3or );
        _mm512_store_pd( out + N4 * 3 + 8, tw3oi );

    }

    inline void butterfly( double * data, const double * twiddle, uint32_t stage_size ) {

        const size_t N4 = stage_size >> 1;

        __m512d re0 = _mm512_load_pd(data         	  );
		__m512d im0 = _mm512_load_pd(data + 8         );
        __m512d re1 = _mm512_load_pd(data 	  + N4    );
		__m512d im1 = _mm512_load_pd(data + 8 + N4    );
        __m512d re2 = _mm512_load_pd(data 	  + N4 * 2);
		__m512d im2 = _mm512_load_pd(data + 8 + N4 * 2);
        __m512d re3 = _mm512_load_pd(data 	  + N4 * 3);
		__m512d im3 = _mm512_load_pd(data + 8 + N4 * 3);

        __m512d sum02re = _mm512_add_pd( re0, re2 );
        __m512d sum02im = _mm512_add_pd( im0, im2 );
        __m512d sum13re = _mm512_add_pd( re1, re3 );
        __m512d sum13im = _mm512_add_pd( im1, im3 );

        _mm512_store_pd(    data,    	_mm512_add_pd(sum02re, sum13re)     );
        _mm512_store_pd(    data + 8, 	_mm512_add_pd(sum02im, sum13im)     );

        __m512d w2r = _mm512_sub_pd(sum02re, sum13re);
        __m512d w2i = _mm512_sub_pd(sum02im, sum13im);

        __m512d tw2r = _mm512_load_pd(twiddle + 2 * vector_width);
        __m512d tw2i = _mm512_load_pd(twiddle + 3 * vector_width);

        __m512d tw2or = _mm512_sub_pd(   _mm512_mul_pd(w2r, tw2r),   _mm512_mul_pd(w2i, tw2i)    );
        __m512d tw2oi = _mm512_add_pd(   _mm512_mul_pd(w2i, tw2r),   _mm512_mul_pd(w2r, tw2i)    );

        _mm512_store_pd(    data + N4,    	tw2or    );
        _mm512_store_pd(    data + 8 + N4, 	tw2oi    );

        __m512d diff02re = _mm512_sub_pd( re0, re2 );
        __m512d diff02im = _mm512_sub_pd( im0, im2 );
        __m512d diff13re = _mm512_sub_pd( re1, re3 );
        __m512d diff13im = _mm512_sub_pd( im1, im3 );

        __m512d tw1r = _mm512_load_pd(twiddle            		);
        __m512d tw1i = _mm512_load_pd(twiddle + vector_width    );

        __m512d tw3r = _mm512_load_pd(twiddle + vector_width * 4);
        __m512d tw3i = _mm512_load_pd(twiddle + vector_width * 5);

        __m512d w3r = _mm512_sub_pd( diff02re, diff13im );
        __m512d w3i = _mm512_add_pd( diff02im, diff13re );
        __m512d w1r = _mm512_add_pd( diff02re, diff13im );
        __m512d w1i = _mm512_sub_pd( diff02im, diff13re );

        __m512d tw1or = _mm512_sub_pd(   _mm512_mul_pd(w1r, tw1r),   _mm512_mul_pd(w1i, tw1i)    );
        __m512d tw1oi = _mm512_add_pd(   _mm512_mul_pd(w1i, tw1r),   _mm512_mul_pd(w1r, tw1i)    );

        __m512d tw3or = _mm512_sub_pd(   _mm512_mul_pd(w3r, tw3r),   _mm512_mul_pd(w3i, tw3i)    );
        __m512d tw3oi = _mm512_add_pd(   _mm512_mul_pd(w3i, tw3r),   _mm512_mul_pd(w3r, tw3i)    );

        _mm512_store_pd(    data + N4 * 2,    	tw1or    );
        _mm512_store_pd(    data + 8 + N4 * 2,  tw1oi    );

        _mm512_store_pd(    data + N4 * 3,    	tw3or    );
        _mm512_store_pd(    data + 8 + N4 * 3,  tw3oi    );

    }

    inline void butterfly_16( double * data, const double * twiddle, uint32_t stage_size ) {

        __m512d xmx = _mm512_load_pd( data      );
        __m512d xmy = _mm512_load_pd( data + 32 );
        __m512d xmu = _mm512_unpacklo_pd( xmx, xmy );       // 0 re
        __m512d xmv = _mm512_unpackhi_pd( xmx, xmy );       // 1 re

                xmx = _mm512_load_pd( data + 16 );
                xmy = _mm512_load_pd( data + 48 );

        __m512d xmz = _mm512_unpacklo_pd( xmx, xmy );       // 2 re

        __m512d xm0 = _mm512_add_pd( xmu, xmz );     // +02 re
        __m512d xm1 = _mm512_sub_pd( xmu, xmz );     // -02 re

                xmz = _mm512_unpackhi_pd( xmx, xmy );       // 3 re

        __m512d xm2 = _mm512_add_pd( xmv, xmz );     // +13 re
        __m512d xm3 = _mm512_sub_pd( xmv, xmz );     // -13 re

                xmx = _mm512_load_pd( data +  8 );
                xmy = _mm512_load_pd( data + 40 );
                xmu = _mm512_unpacklo_pd( xmx, xmy );       // 0 im
                xmv = _mm512_unpackhi_pd( xmx, xmy );       // 1 im

                xmx = _mm512_load_pd( data + 24 );
                xmy = _mm512_load_pd( data + 56 );

                xmz = _mm512_unpacklo_pd( xmx, xmy );       // 2 im

        __m512d xm4 = _mm512_add_pd( xmu, xmz );     // +02 im
        __m512d xm5 = _mm512_sub_pd( xmu, xmz );     // -02 im

                xmz = _mm512_unpackhi_pd( xmx, xmy );       // 3 im

        __m512d xm6 = _mm512_add_pd( xmv, xmz );     // +13 im
        __m512d xm7 = _mm512_sub_pd( xmv, xmz );     // -13 im

        __m512d w0r = _mm512_add_pd( xm0, xm2 );
        __m512d w0i = _mm512_add_pd( xm4, xm6 );

        __m512d w2r = _mm512_sub_pd( xm0, xm2 );
        __m512d w2i = _mm512_sub_pd( xm4, xm6 );

        __m512d tw2r = _mm512_load_pd( twiddle + vector_width * 2 );
        __m512d tw2i = _mm512_load_pd( twiddle + vector_width * 3 );

        __m512d tw2or = _mm512_sub_pd( _mm512_mul_pd(w2r, tw2r), _mm512_mul_pd(w2i, tw2i));
        __m512d tw2oi = _mm512_add_pd( _mm512_mul_pd(w2i, tw2r), _mm512_mul_pd(w2r, tw2i));

        __m512d tw1r = _mm512_load_pd(twiddle                    );
        __m512d tw1i = _mm512_load_pd(twiddle + vector_width     );

        __m512d tw3r = _mm512_load_pd(twiddle + vector_width * 4 );
        __m512d tw3i = _mm512_load_pd(twiddle + vector_width * 5 );

        __m512d w3r = _mm512_sub_pd( xm1, xm7 );
        __m512d w3i = _mm512_add_pd( xm5, xm3 );
        __m512d w1r = _mm512_add_pd( xm1, xm7 );
        __m512d w1i = _mm512_sub_pd( xm5, xm3 );

        __m512d tw1or = _mm512_sub_pd( _mm512_mul_pd(w1r, tw1r), _mm512_mul_pd(w1i, tw1i));
        __m512d tw1oi = _mm512_add_pd( _mm512_mul_pd(w1i, tw1r), _mm512_mul_pd(w1r, tw1i));

        __m512d tw3or = _mm512_sub_pd( _mm512_mul_pd(w3r, tw3r), _mm512_mul_pd(w3i, tw3i));
        __m512d tw3oi = _mm512_add_pd( _mm512_mul_pd(w3i, tw3r), _mm512_mul_pd(w3r, tw3i));

        _mm512_store_pd(data     , _mm512_unpacklo_pd(w0r,tw2or));
        _mm512_store_pd(data + 8 , _mm512_unpacklo_pd(w0i,tw2oi));
        _mm512_store_pd(data + 16, _mm512_unpacklo_pd(tw1or,tw3or));
        _mm512_store_pd(data + 24, _mm512_unpacklo_pd(tw1oi,tw3oi));
        _mm512_store_pd(data + 32, _mm512_unpackhi_pd(w0r,tw2or));
        _mm512_store_pd(data + 40, _mm512_unpackhi_pd(w0i,tw2oi));
        _mm512_store_pd(data + 48, _mm512_unpackhi_pd(tw1or,tw3or));
        _mm512_store_pd(data + 56, _mm512_unpackhi_pd(tw1oi,tw3oi));

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

        _MM512_FAKE_TRANSPOSE_4x8_PD(re0,re1,re2,re3);
        _MM512_FAKE_TRANSPOSE_4x8_PD(im0,im1,im2,im3);

        __m512d sum02re = _mm512_add_pd( re0, re2 );
        __m512d sum02im = _mm512_add_pd( im0, im2 );
        __m512d sum13re = _mm512_add_pd( re1, re3 );
        __m512d sum13im = _mm512_add_pd( im1, im3 );

        __m512d w0re = _mm512_add_pd(sum02re, sum13re);
        __m512d w0im = _mm512_add_pd(sum02im, sum13im);
        __m512d w2re = _mm512_sub_pd(sum02re, sum13re);
        __m512d w2im = _mm512_sub_pd(sum02im, sum13im);

        __m512d diff02re = _mm512_sub_pd( re0, re2 );
        __m512d diff02im = _mm512_sub_pd( im0, im2 );
        __m512d diff13re = _mm512_sub_pd( re1, re3 );
        __m512d diff13im = _mm512_sub_pd( im1, im3 );

        __m512d w3re = _mm512_sub_pd( diff02re, diff13im );
        __m512d w3im = _mm512_add_pd( diff02im, diff13re );
        __m512d w1re = _mm512_add_pd( diff02re, diff13im );
        __m512d w1im = _mm512_sub_pd( diff02im, diff13re );

        _MM512_FAKE_TRANSPOSE_B_4x8_PD(w0re,w1re,w2re,w3re);
        _MM512_FAKE_TRANSPOSE_B_4x8_PD(w0im,w1im,w2im,w3im);

        _mm512_store_pd(data     , _mm512_unpacklo_pd(w0re,w0im));
        _mm512_store_pd(data + 8 , _mm512_unpackhi_pd(w0re,w0im));
        _mm512_store_pd(data + 16, _mm512_unpacklo_pd(w1re,w1im));
        _mm512_store_pd(data + 24, _mm512_unpackhi_pd(w1re,w1im));
        _mm512_store_pd(data + 32, _mm512_unpacklo_pd(w2re,w2im));
        _mm512_store_pd(data + 40, _mm512_unpackhi_pd(w2re,w2im));
        _mm512_store_pd(data + 48, _mm512_unpacklo_pd(w3re,w3im));
        _mm512_store_pd(data + 56, _mm512_unpackhi_pd(w3re,w3im));

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
		if ( TOP(ITER)++ < 4 ) return true;
		return false;
	}

	inline void fft_recursive( double * data, double * twiddle_data ) {

		st_push( size >> 2, data, twiddle_data );

		while ( st_ptr ) { // stack not empty

			if ( st_next() ) {

				if ( TOP(SIZE) == 1024 ) {

					double * subdata = TOP_(DATA);
					double * subtwiddle = TOP_(TWID);

					// 1024 unroll

						for ( size_t n = 0; n < 256; n += vector_width ) {

							butterfly( subdata, subtwiddle + n * 6, 1024 );
							subdata += vector_next;
						}

					subtwiddle += 1536;

					// 256 unroll

					subdata = TOP_(DATA);

					for ( size_t i = 0; i < 1024; i += 256 ) {

						for ( size_t n = 0; n < 64; n += vector_width ) {

							butterfly( subdata, subtwiddle + n * 6, 256 );
							subdata += vector_next;
						}
						subdata += 384;
					}
					subtwiddle += 384;

					// 64 unroll

					subdata = TOP_(DATA);

					for ( size_t i = 0; i < 1024; i += 64 ) {

						for ( size_t n = 0; n < 16; n += vector_width ) {

							butterfly( subdata, subtwiddle + n * 6, 64 );
							subdata += vector_next;
						}
						subdata += 96;
					}
					subtwiddle += 96;

					// 16 unroll

					subdata = TOP_(DATA);

					for ( size_t i = 0; i < 1024; i += 32 ) {

						butterfly_16( subdata, subtwiddle, 16 );
						subdata += 64;
					}

					// 4 unroll

					subdata = TOP_(DATA);

					for ( size_t n = 0; n < 1024; n += 32 ) {

						butterfly_final( subdata );
						subdata += 64;
					}

					TOP_(DATA) += 2048;

				} else {

					const size_t N4 = TOP(SIZE) >> 2;
					double * safe = TOP_(DATA);

					for ( size_t n = 0; n < N4; n += vector_width ) {

						butterfly( TOP_(DATA), TOP_(TWID) + n * 6, TOP(SIZE) );
						TOP_(DATA) += vector_next;
					}
					TOP_(DATA) += 6 * N4;

					st_push( N4, safe, TOP_(TWID) + 6 * N4 );
				}

			} else st_pop();
		}
	}

    void fft( const double * in, double * out ) {

		double * data         = out;
		double * twiddle_data = twiddle;

        // depth 0 with vec reorder re-im-re-im-re-im-re-im   to   re-re-re-re-im-im-im-im
        {
            const size_t N4                           = size >> 2;

            for ( size_t n = 0; n < N4; n += vector_width ) {

                butterfly_zero( in, data, twiddle_data + n * 6, size );
                in   += vector_next;
                data += vector_next;
            }
            twiddle_data += 6 * N4; // 2 * 3 * N4
        }

		data = out;

		if ( depth > 5 ) fft_recursive( data, twiddle_data ); // unroll 1024

		else {

			size_t step = size >> 2;
			const size_t top = depth - 2;

			  for ( size_t k = 1; k < top; ++k, step >>= 2 ) {

			      data = out;

			      const size_t N4 = step >> 2;

			      for ( size_t i = 0; i < size; i += step ) {

			          for ( size_t n = 0; n < N4; n += vector_width ) {

			              butterfly( data, twiddle_data + n * 6, step );
			              data += vector_next;
			          }
			          data += 6 * N4;
			      }
			      twiddle_data += 6 * step >> 2;
			  }

            data = out;

            for ( size_t i = 0; i < size; i += 32 ) {

                butterfly_16( data, twiddle_data, 16 );
                data += 64;
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
        if (i%8==0) std::cout << std::endl;
		std::cout <<std::fixed<< std::setprecision(2) << output[i*2] << " " << output[i*2+1] << "i  " ;
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

	if ( base == 4 ) {

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
    for ( int i = 3; i < 13; ++i ) time2n<double>(i,4);
}

int main(void)
{
		//*/
		test_output(64);
		test_output(1024);
		test_output(4096);
		test_speed();
		/*/
    	test_speed();
		//*/
	return 0;
}
