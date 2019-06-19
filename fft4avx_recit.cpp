#include <bits/stdc++.h>
#include <immintrin.h>
#include "bitrev_complex.hpp"

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

				for ( size_t s = size; s > 4; s /= 4 ) twiddle_size |= s;

        twiddle = (double*) aligned_alloc( native_alignment, 2 * twiddle_size * sizeof(double) );

        double * twiddleinit = twiddle;

        depth = 1;

        for ( size_t stage_size = size, blocks = 1; stage_size > 4; stage_size >>= 2, blocks <<= 2 ) {

        	depth++;

            for ( size_t n = 0; n < stage_size >> 2; n += vector_width ) {

                initialize_twiddles( twiddleinit, n * blocks    , blocks    , size );
                initialize_twiddles( twiddleinit, n * blocks * 2, blocks * 2, size );
                initialize_twiddles( twiddleinit, n * blocks * 3, blocks * 3, size );
            }
        }
    }

    void fft_free ( void ) {

        free(twiddle);

    }

    inline void butterfly_zero( const double * in, double * out, const double * twiddle, uint32_t stage_size ) {

        const size_t N4 = stage_size >> 1;

        // 0 ..N4| 1 ..2N4| 2 ..3N4| 3

        __m256d xmx = _mm256_load_pd( in 	          	);
        __m256d xmy = _mm256_load_pd( in          + 4 );
        __m256d xm4 = _mm256_unpacklo_pd( xmx, xmy );       // 0 re
        __m256d xm5 = _mm256_unpackhi_pd( xmx, xmy );       // 0 im

                xmx = _mm256_load_pd( in + N4 * 2     );
                xmy = _mm256_load_pd( in + N4 * 2 + 4 );

        __m256d xmz = _mm256_unpacklo_pd( xmx, xmy );

        __m256d xm0 = _mm256_add_pd( xm4, xmz );     // +02 re
                xm4 = _mm256_sub_pd( xm4, xmz );     // -02 re

                xmz = _mm256_unpackhi_pd( xmx, xmy );
        __m256d xm1 = _mm256_add_pd( xm5, xmz );     // +02 im
                xm5 = _mm256_sub_pd( xm5, xmz );     // -02 im

                xmx = _mm256_load_pd( in + N4         );
                xmy = _mm256_load_pd( in + N4     + 4 );
        __m256d xm6 = _mm256_unpacklo_pd( xmx, xmy );       // 1 re
        __m256d xm7 = _mm256_unpackhi_pd( xmx, xmy );       // 1 im


                xmx = _mm256_load_pd( in + N4 * 3     );
                xmy = _mm256_load_pd( in + N4 * 3 + 4 );

                xmz = _mm256_unpacklo_pd( xmx, xmy );
        __m256d xm2 = _mm256_add_pd( xm6, xmz );     // +13 re
                xm6 = _mm256_sub_pd( xm6, xmz );     // -13 re

                xmz = _mm256_unpackhi_pd( xmx, xmy );
        __m256d xm3 = _mm256_add_pd( xm7, xmz );     // +13 im
                xm7 = _mm256_add_pd( xm7, xmz );     // -13 im


        __m256d w2r = _mm256_sub_pd( xm0, xm2 );
        __m256d w2i = _mm256_sub_pd( xm1, xm3 );

        __m256d tw2r = _mm256_load_pd( twiddle + vector_width * 2 );
        __m256d tw2i = _mm256_load_pd( twiddle + vector_width * 3 );

        __m256d tw2or = _mm256_sub_pd( _mm256_mul_pd(w2r, tw2r), _mm256_mul_pd(w2i, tw2i));
        __m256d tw2oi = _mm256_add_pd( _mm256_mul_pd(w2i, tw2r), _mm256_mul_pd(w2r, tw2i));

        __m256d tw1r = _mm256_load_pd(twiddle                    );
        __m256d tw1i = _mm256_load_pd(twiddle + vector_width     );

        __m256d tw3r = _mm256_load_pd(twiddle + vector_width * 4 );
        __m256d tw3i = _mm256_load_pd(twiddle + vector_width * 5 );

        __m256d w3r = _mm256_sub_pd( xm4, xm7 );
        __m256d w3i = _mm256_add_pd( xm5, xm6 );
        __m256d w1r = _mm256_add_pd( xm4, xm7 );
        __m256d w1i = _mm256_sub_pd( xm5, xm6 );

        __m256d tw1or = _mm256_sub_pd( _mm256_mul_pd(w1r, tw1r), _mm256_mul_pd(w1i, tw1i));
        __m256d tw1oi = _mm256_add_pd( _mm256_mul_pd(w1i, tw1r), _mm256_mul_pd(w1r, tw1i));

        __m256d tw3or = _mm256_sub_pd( _mm256_mul_pd(w3r, tw3r), _mm256_mul_pd(w3i, tw3i));
        __m256d tw3oi = _mm256_add_pd( _mm256_mul_pd(w3i, tw3r), _mm256_mul_pd(w3r, tw3i));

        _mm256_store_pd( out             , _mm256_add_pd( xm0, xm2 ));
        _mm256_store_pd( out          + 4, _mm256_add_pd( xm1, xm3 ));

        _mm256_store_pd( out + N4        , tw2or );
        _mm256_store_pd( out + N4     + 4, tw2oi );

        _mm256_store_pd( out + N4 * 2    , tw1or );
        _mm256_store_pd( out + N4 * 2 + 4, tw1oi );

        _mm256_store_pd( out + N4 * 3    , tw3or );
        _mm256_store_pd( out + N4 * 3 + 4, tw3oi );

    }

    inline void butterfly( double * data, const double * twiddle, uint32_t stage_size ) {

        const size_t N4 = stage_size >> 1;

        __m256d re0 = _mm256_load_pd(data         		);
				__m256d im0 = _mm256_load_pd(data + 4         );
        __m256d re1 = _mm256_load_pd(data 		+ N4    );
				__m256d im1 = _mm256_load_pd(data + 4 + N4    );
        __m256d re2 = _mm256_load_pd(data 		+ N4 * 2);
				__m256d im2 = _mm256_load_pd(data + 4 + N4 * 2);
        __m256d re3 = _mm256_load_pd(data 		+ N4 * 3);
				__m256d im3 = _mm256_load_pd(data + 4 + N4 * 3);

        __m256d sum02re = _mm256_add_pd( re0, re2 );
        __m256d sum02im = _mm256_add_pd( im0, im2 );
        __m256d sum13re = _mm256_add_pd( re1, re3 );
        __m256d sum13im = _mm256_add_pd( im1, im3 );

        _mm256_store_pd(    data,    _mm256_add_pd(sum02re, sum13re)     );
        _mm256_store_pd(    data + 4,    _mm256_add_pd(sum02im, sum13im)     );

        __m256d w2r = _mm256_sub_pd(sum02re, sum13re);
        __m256d w2i = _mm256_sub_pd(sum02im, sum13im);

        __m256d tw2r = _mm256_load_pd(twiddle + 2 * vector_width);
        __m256d tw2i = _mm256_load_pd(twiddle + 3 * vector_width);

        __m256d tw2or = _mm256_sub_pd(   _mm256_mul_pd(w2r, tw2r),   _mm256_mul_pd(w2i, tw2i)    );
        __m256d tw2oi = _mm256_add_pd(   _mm256_mul_pd(w2i, tw2r),   _mm256_mul_pd(w2r, tw2i)    );

        _mm256_store_pd(    data + N4    ,    tw2or    );
        _mm256_store_pd(    data + 4 + N4    ,    tw2oi    );

        __m256d diff02re = _mm256_sub_pd( re0, re2 );
        __m256d diff02im = _mm256_sub_pd( im0, im2 );
        __m256d diff13re = _mm256_sub_pd( re1, re3 );
        __m256d diff13im = _mm256_sub_pd( im1, im3 );

        __m256d tw1r = _mm256_load_pd(twiddle            );
        __m256d tw1i = _mm256_load_pd(twiddle + vector_width    );

        __m256d tw3r = _mm256_load_pd(twiddle + vector_width * 4);
        __m256d tw3i = _mm256_load_pd(twiddle + vector_width * 5);

        __m256d w3r = _mm256_sub_pd( diff02re, diff13im );
        __m256d w3i = _mm256_add_pd( diff02im, diff13re );
        __m256d w1r = _mm256_add_pd( diff02re, diff13im );
        __m256d w1i = _mm256_sub_pd( diff02im, diff13re );

        __m256d tw1or = _mm256_sub_pd(   _mm256_mul_pd(w1r, tw1r),   _mm256_mul_pd(w1i, tw1i)    );
        __m256d tw1oi = _mm256_add_pd(   _mm256_mul_pd(w1i, tw1r),   _mm256_mul_pd(w1r, tw1i)    );

        __m256d tw3or = _mm256_sub_pd(   _mm256_mul_pd(w3r, tw3r),   _mm256_mul_pd(w3i, tw3i)    );
        __m256d tw3oi = _mm256_add_pd(   _mm256_mul_pd(w3i, tw3r),   _mm256_mul_pd(w3r, tw3i)    );

        _mm256_store_pd(    data + N4 * 2,    tw1or    );
        _mm256_store_pd(    data + 4 + N4 * 2,    tw1oi    );

        _mm256_store_pd(    data + N4 * 3,    tw3or    );
        _mm256_store_pd(    data + 4 + N4 * 3,    tw3oi    );

    }

    inline void butterfly_final( double * data ) {

        __m256d re0 = _mm256_load_pd(data     );
				__m256d im0 = _mm256_load_pd(data + 4 );
        __m256d re1 = _mm256_load_pd(data + 8 );
				__m256d im1 = _mm256_load_pd(data + 12);
        __m256d re2 = _mm256_load_pd(data + 16);
				__m256d im2 = _mm256_load_pd(data + 20);
        __m256d re3 = _mm256_load_pd(data + 24);
        __m256d im3 = _mm256_load_pd(data + 28);

        _MM256_TRANSPOSE_4x4_PD(re0,re1,re2,re3);
        _MM256_TRANSPOSE_4x4_PD(im0,im1,im2,im3);

        /* in next sum and diff we swap 1 and 2 of both re and im by placing them to opposite operation (speedup) */

        __m256d sum02re = _mm256_add_pd( re0, re1 );    // should be re0, re2 but swapped to re0, re1; everything including 1 or 2 does the same thing
        __m256d sum02im = _mm256_add_pd( im0, im1 );
        __m256d sum13re = _mm256_add_pd( re2, re3 );
        __m256d sum13im = _mm256_add_pd( im2, im3 );

        __m256d w0re = _mm256_add_pd(sum02re, sum13re);
        __m256d w0im = _mm256_add_pd(sum02im, sum13im);
        __m256d w2re = _mm256_sub_pd(sum02re, sum13re);
        __m256d w2im = _mm256_sub_pd(sum02im, sum13im);

        __m256d diff02re = _mm256_sub_pd( re0, re1 );
        __m256d diff02im = _mm256_sub_pd( im0, im1 );
        __m256d diff13re = _mm256_sub_pd( re2, re3 );
        __m256d diff13im = _mm256_sub_pd( im2, im3 );

        __m256d w3re = _mm256_sub_pd( diff02re, diff13im );
        __m256d w3im = _mm256_add_pd( diff02im, diff13re );
        __m256d w1re = _mm256_add_pd( diff02re, diff13im );
        __m256d w1im = _mm256_sub_pd( diff02im, diff13re );

        _MM256_TRANSPOSE_4x4_PD(w0re,w1re,w2re,w3re);
        _MM256_TRANSPOSE_4x4_PD(w0im,w1im,w2im,w3im);

        _mm256_store_pd(data     , _mm256_unpacklo_pd(w0re,w0im));
        _mm256_store_pd(data + 4 , _mm256_unpackhi_pd(w0re,w0im));
        _mm256_store_pd(data + 8 , _mm256_unpacklo_pd(w1re,w1im));
        _mm256_store_pd(data + 12, _mm256_unpackhi_pd(w1re,w1im));
        _mm256_store_pd(data + 16, _mm256_unpacklo_pd(w2re,w2im));
        _mm256_store_pd(data + 20, _mm256_unpackhi_pd(w2re,w2im));
        _mm256_store_pd(data + 24, _mm256_unpacklo_pd(w3re,w3im));
        _mm256_store_pd(data + 28, _mm256_unpackhi_pd(w3re,w3im));

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

					for ( size_t i = 0; i < 1024; i += 16 ) {

						butterfly( subdata, subtwiddle, 16 );
						subdata += 32;
					}

					// 4 unroll

					subdata = TOP_(DATA);

					for ( size_t n = 0; n < 1024; n += 16 ) {

						butterfly_final( subdata );
						subdata += 32;
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
			const size_t top = depth - 1;

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

		    for ( size_t n = 0; n < size; n += 16 ) {

		        butterfly_final( data );
		        data += 32;
		    }

		}

		//revbin_permute( out, size );

    }

};

/*void test_output( size_t N ) {

	double * input = (double*)aligned_alloc(native_alignment, N * 2 * sizeof(double));
	double * output = (double*)aligned_alloc(native_alignment, N * 2 * sizeof(double));
	for ( uint32_t i = 0; i < N; ++i ) {
		input[i*2]   = (double)i;
		input[i*2+1] = (double)0.;
	}

	fft_plan FFT( N );
	FFT.fft ( input, output );
	FFT.fft_free();


	for ( uint32_t i = 0; i < 8; ++i ) {
		std::cout <<std::fixed<< std::setprecision(2) << output[i*2] << " " << output[i*2+1] << "i  " ;
	}

	std::cout << std::endl;

}*/

template <typename T>
constexpr inline std::uint32_t log_2( T n )
{
    return (n > 1) ? 1 + log_2(n >> 1) : 0;
}

template <typename T, uint32_t n>
void time2n( uint32_t tests = 10 ) {

    uint32_t m = 1 << n;

    using namespace std::chrono;

    std::vector<double> mflops(tests);
    std::vector<duration<double> > time(tests);

    T * input  = (T*)aligned_alloc(native_alignment, 2 * m * sizeof(T));
    T * output = (T*)aligned_alloc(native_alignment, 2 * m * sizeof(T));

    fft_plan FFT( m );
    BitRev<T, 6, n> br;
    T * cb = br.allocate();

    for ( uint32_t k = 0; k < tests; ++k ) {

        for ( uint32_t i = 0; i < m; ++i ) {
            input[i*2]   = (T)0.;
            input[i*2+1] = (T)0.;
        }

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        FFT.fft ( input, output );

        br.reverse(input, cb);

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

        time[k] = time_span;

        // 5 * log_2(n) / time

        mflops[k] = 5 * m * log_2(m) / (std::log(2.0)*time_span.count() * 1000000.0F);

    }

    FFT.fft_free();
    free(cb);
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


int main(void)
{

        std::cout<<"#size"<<'\t'<<"speed(mflops)"<<'\t'<<"time(ms) r4"<<std::endl;

        time2n<double,24>();

	return 0;
}
