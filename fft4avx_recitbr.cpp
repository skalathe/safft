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

// core of bitreversal from https://bitbucket.org/orserang/bit-reversal-methods/

template <unsigned char LOG_N>
class BitReversal {
protected:
  static const unsigned char reversed_byte_table[256];

public:
  // Acknowledgment goes to Sean Eron Anderson's Bit Twiddling Hacks page:
  // graphics.stanford.edu/~seander/bithacks.html
  inline static unsigned int reverse_int_logical(unsigned int x) {
    // swap odd and even bits
    x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
    // swap consecutive pairs
    x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
    // swap nibbles ...
    x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
    // swap bytes
    x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);
    // swap 2-byte long pairs
    x = ( x >> 16             ) | ( x               << 16);
    return x;
  }

  // Adapted from Sean Eron Anderson:
  inline static unsigned short reverse_short_byte_table(unsigned short x){
    unsigned char inByte0 = (x & 0xFF);
    unsigned char inByte1 = (x & 0xFF00) >> 8;
    return (reversed_byte_table[inByte0] << 8) | reversed_byte_table[inByte1];
  }

  // Adapted from Sean Eron Anderson:
  inline static unsigned int reverse_int_byte_table(unsigned int x){
    unsigned char inByte0 = (x & 0xFF);
    unsigned char inByte1 = (x & 0xFF00) >> 8;
    unsigned char inByte2 = (x & 0xFF0000) >> 16;
    unsigned char inByte3 = (x & 0xFF000000) >> 24;
    return (reversed_byte_table[inByte0] << 24) | (reversed_byte_table[inByte1] << 16) | (reversed_byte_table[inByte2] << 8) | reversed_byte_table[inByte3];
  }

  // Adapted from Sean Eron Anderson:
  inline static unsigned long reverse_long_byte_table(unsigned long x){
    unsigned char inByte0 = (x & 0xFF);
    unsigned char inByte1 = (x & 0xFF00) >> 8;
    unsigned char inByte2 = (x & 0xFF0000) >> 16;
    unsigned char inByte3 = (x & 0xFF000000) >> 24;
    unsigned char inByte4 = (x & 0xFF00000000ul) >> 32;
    unsigned char inByte5 = (x & 0xFF0000000000ul) >> 40;
    unsigned char inByte6 = (x & 0xFF000000000000ul) >> 48;
    unsigned char inByte7 = (x & 0xFF00000000000000ul) >> 56;
    return ((unsigned long)reversed_byte_table[inByte0] << 56) | ((unsigned long)reversed_byte_table[inByte1] << 48) | ((unsigned long)reversed_byte_table[inByte2] << 40) | ((unsigned long)reversed_byte_table[inByte3] << 32) | (reversed_byte_table[inByte4] << 24) | (reversed_byte_table[inByte5] << 16) | (reversed_byte_table[inByte6] << 8) | reversed_byte_table[inByte7];
  }

  // From Sean Eron Anderson:
  inline static unsigned long reverse_bitwise(unsigned long x) {
    unsigned long maskFromLeft = 1<<LOG_N;
    unsigned long res = 0;
    unsigned int bitNum = LOG_N;
    while (maskFromLeft > 0) {
      unsigned char bit = (x & maskFromLeft) >> bitNum;
      res |= ( bit << (LOG_N-1-bitNum) );
      --bitNum;
      maskFromLeft >>= 1;
    }
    return res;
  }

  inline static unsigned long reverse_bytewise(unsigned long x) {
    // if (constexpr) statements should be eliminated by compiler to
    // choose correct case at compile time:

    if (LOG_N > sizeof(unsigned int)*8) {
      // Work with long reversal:

      // sizeof(unsigned long) * 8:
      //      const unsigned int bitsPerLong = sizeof(unsigned long)<<3;

      // Pure bit reversal of 1 will result in 1<<63; need to shift
      // right so that reversal of 1 yields LOG_N:
      return reverse_long_byte_table(x) >> (sizeof(unsigned long)*8 - LOG_N);
    }
    else if (LOG_N > sizeof(unsigned short)*8) {
      // Work with int reversal:

      // sizeof(unsigned int) * 8:
      //    const unsigned int bitsPerInt = sizeof(unsigned int)<<3;
      // Pure bit reversal of 1 will result in 1<<31; need to shift
      // right so that reversal of 1 yields LOG_N:
      return reverse_int_byte_table(x) >> (sizeof(unsigned int)*8 - LOG_N);
    }
    else if (LOG_N > sizeof(unsigned char)*8) {
      // Work with short int reversal:
      return reverse_short_byte_table(x) >> (sizeof(unsigned short)*8 - LOG_N);
    }
    // Work with char reversal:
    return reversed_byte_table[x] >> (sizeof(unsigned char)*8 - LOG_N);
  }

  inline static unsigned int reverse_bytewise(unsigned int x) {
    // To prevent unnecessary warnings about (desired) behavior when LOG_N == 0:
    if (LOG_N == 0)
      return x;
    return reverse_int_byte_table(x) >> (sizeof(unsigned int)*8 - LOG_N);
  }

  // Using XOR recurrence (simple homemade method from Oliver Serang):
  inline static void advance_index_and_reversed(unsigned long & index, unsigned long & reversed) {
    unsigned long temp = index+1;
    unsigned long tail = ( index ^ temp );
    // tail is of the form 00...011...1

    index = temp;
    // create the reverse of tail, which is of form 11...100...0:
    auto shift = __builtin_clzl(tail);
    tail <<= shift;
    tail >>= ((sizeof(unsigned long)*8)-LOG_N);

    // xor reversed with reversed tail gives reversed of index+1:
    reversed ^= tail;
  }
};

// From StackOverflow:
// http://stackoverflow.com/questions/746171/best-algorithm-for-bit-reversal-from-msb-lsb-to-lsb-msb-in-c#24058332
template<unsigned char LOG_N>
const unsigned char BitReversal<LOG_N>::reversed_byte_table[256] = {0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA, 0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE, 0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9, 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD, 0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB, 0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

template<typename T, unsigned char LOG_N, unsigned char LOG_BLOCK_WIDTH>
class COBRAShuffle {
public:
  inline static void apply(T*__restrict const v) {
    constexpr unsigned char NUM_B_BITS = LOG_N - 2*LOG_BLOCK_WIDTH;
    constexpr unsigned long B_SIZE = 1ul << NUM_B_BITS;
    constexpr unsigned long BLOCK_WIDTH = 1ul << LOG_BLOCK_WIDTH;

    T*__restrict buffer = (T*)malloc(sizeof(T)*BLOCK_WIDTH*BLOCK_WIDTH*2);

    for (unsigned long b=0; b<B_SIZE; ++b) {
      unsigned long b_rev = BitReversal<NUM_B_BITS>::reverse_bytewise(b);

      // Copy block to buffer:
      for (unsigned long a=0; a<BLOCK_WIDTH; ++a) {
	unsigned long a_rev = BitReversal<LOG_BLOCK_WIDTH>::reverse_bytewise(a);

	for (unsigned long c=0; c<BLOCK_WIDTH; ++c) {
	  	  		buffer[( (a_rev << LOG_BLOCK_WIDTH) | c )*2] = v[( (a << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c )*2];
				buffer[( (a_rev << LOG_BLOCK_WIDTH) | c )*2+1] = v[( (a << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c )*2+1];
	}
      }

      // Swap v[rev_index] with buffer:
      for (unsigned long c=0; c<BLOCK_WIDTH; ++c) {
      	// Note: Typo in original pseudocode by Carter and Gatlin at
      	// the following line:
      	unsigned long c_rev = BitReversal<LOG_BLOCK_WIDTH>::reverse_bytewise(c);

      	for (unsigned long a_rev=0; a_rev<BLOCK_WIDTH; ++a_rev) {
      	  unsigned long a = BitReversal<LOG_BLOCK_WIDTH>::reverse_bytewise(a_rev);
      	  // To guarantee each value is swapped only one time:
      	  // index < reversed_index <-->
      	  // a b c < c' b' a' <-->
      	  // a < c' ||
      	  // a <= c' && b < b' ||
      	  // a <= c' && b <= b' && a' < c

	  bool index_less_than_reverse = a < c_rev || (a == c_rev && b < b_rev) || (a == c_rev && b == b_rev && a_rev < c);
	  if ( index_less_than_reverse ) {
				std::swap( v[((c_rev << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b_rev<<LOG_BLOCK_WIDTH) | a_rev)*2], buffer[( (a_rev<<LOG_BLOCK_WIDTH) | c )*2]);
				std::swap( v[((c_rev << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b_rev<<LOG_BLOCK_WIDTH) | a_rev)*2+1], buffer[( (a_rev<<LOG_BLOCK_WIDTH) | c )*2+1]);
	}
	}
      }

      // Copy changes that were swapped into buffer above:
      for (unsigned long a=0; a<BLOCK_WIDTH; ++a) {
	unsigned long a_rev = BitReversal<LOG_BLOCK_WIDTH>::reverse_bytewise(a);
	for (unsigned long c=0; c<BLOCK_WIDTH; ++c) {
	  unsigned long c_rev = BitReversal<LOG_BLOCK_WIDTH>::reverse_bytewise(c);
	  bool index_less_than_reverse = a < c_rev || (a == c_rev && b < b_rev) || (a == c_rev && b == b_rev && a_rev < c);

	  if (index_less_than_reverse) {
				std::swap(v[( (a << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c )*2], buffer[( (a_rev << LOG_BLOCK_WIDTH) | c )*2]);
				std::swap(v[( (a << NUM_B_BITS << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c )*2+1], buffer[( (a_rev << LOG_BLOCK_WIDTH) | c )*2+1]);
				}
	}
      }
    }
    free(buffer);
  }

};

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
		//std::cout << x << " " << r << std::endl;
		//std::cout << r-x << std::endl;
        x = x + 1;
        // x is even
        for ( uint32_t m = n >> 1; (!((r ^= m) & m)); m >>= 1 );

        if ( r > x ) {
            std::swap(data[x*2],data[r*2]);
			std::swap(data[x*2+1],data[r*2+1]);
            std::swap(data[(n-1-x)*2],data[(n-1-r)*2]);
            std::swap(data[(n-1-x)*2+1],data[(n-1-r)*2+1]);
			//std::cout << x << " " << r << " " << (n-1-x) << " " << (n-1-r) << "y" << std::endl;
			//std::cout << r-x << " " << (n-1-x) - (n-1-r) << std::endl;
        } //else std::cout << x << " " << r << " " << (n-1-x) << " " << (n-1-r) << "n" << std::endl;
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

		/*/
			revbin_permute( out, size );
		/*/

    }

};





template <size_t n, size_t N, size_t b>
void test_output() {

	double * input = (double*)aligned_alloc(native_alignment, N * 2 * sizeof(double));
	double * output = (double*)aligned_alloc(native_alignment, N * 2 * sizeof(double));
	double * output2 = (double*)aligned_alloc(native_alignment, N * 2 * sizeof(double));
	for ( uint32_t i = 0; i < N; ++i ) {
		input[i*2]   = (double)i;
		input[i*2+1] = (double)0.;
	}

	fft_plan FFT( N );
	FFT.fft ( input, output );
	memcpy(output2,output,N*2*sizeof(double));
	COBRAShuffle<double,n,b> bitrev;
	bitrev.apply(output);
	revbin_permute(output2,N);
	FFT.fft_free();


	for ( uint32_t i = 0; i < 8; ++i ) {
		std::cout <<std::fixed<< std::setprecision(2) << output[i*2] << " " << output[i*2+1] << "i  " ;
	} std::cout << std::endl;
	bool ok = true;
	for ( uint32_t i = 0; i < N * 2; ++i ) {
		if ( std::abs(output[i] - output2[i]) > 0.0000000001 ) {
			std::cout << output[i] << " != " << output2[i] <<std::endl;
			ok = false;
			break;
		}
	}
	std::cout << (ok?"Good":"Fail") << std::endl;


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

template <typename T, size_t n, size_t m, size_t b>
void time2n( uint32_t base, uint32_t tests = 10 ) {

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

			COBRAShuffle<double,n,b> bitrev;

		    high_resolution_clock::time_point t1 = high_resolution_clock::now();

		    FFT.fft ( input, output );
			bitrev.apply(output);

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

/*void test_speed ( void ) {
    std::cout<<"#size"<<'\t'<<"speed(mflops)"<<'\t'<<"time(ms) r4"<<std::endl;
    for ( int i = 3; i < 13; ++i ) time2n<double>(i,4);
}*/

template <size_t n, size_t N>
void testcache() {
	time2n<double,n, N, 6>(4);
	test_output<n, N, 6>();
	time2n<double,n, N, 7>(4);
	test_output<n, N, 7>();
	/*time2n<double,n, N, 5>(4);
	test_output<n, N, 5>();
	time2n<double,n, N, 4>(4);
	test_output<n, N, 4>();
	time2n<double,n, N, 3>(4);
	test_output<n, N, 3>();
	time2n<double,n, N, 2>(4);
	test_output<n, N, 2>();
	time2n<double,n, N, 1>(4);
	test_output<n, N, 1>();*/
}

int main(void)
{
		/*/
		test_output(64);
		//test_output(1024);
		//test_output(4096);
		/*/
    //test_speed();
	testcache<24, 4096*4096>();
		//*/
	return 0;
}
