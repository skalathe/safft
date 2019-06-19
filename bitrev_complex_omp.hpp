#include <functional>
#include <omp.h>

/*************************************************************************************
    Author: Adam Simek
    Czech Technical University - Faculty of Information Technologies

    Bit Reversal Algorithm based on COBRA

    + Fast (Almost twice as fast as original inplace COBRA)
    + Cache effective
    + Compile time support, SFINAE compile branching
    + Pattern supports AVX, AVX-512 with -O3 automatic vectorization

    - Needs at least C++14
    - Works only with sizes of powers of 2
    - Limited LOG_B * 2 <= LOG_N <= LOG_B * 4 (LOG_B = side of COBRA table, LOG_N = log_2 of problem size N)
*************************************************************************************/

using unsigned_t = unsigned; // specify type of unsigned values
using signed_t   = int;      // specify type of signed values

class BitReversal {

/*
    Reverse byte inspired by Sean Eron Anderson
    http://graphics.stanford.edu/~seander/bithacks.html

    Modified to compile time and SFINAE optimalization

    Any bitreversal would work, just needs to be compile time
*/

    constexpr static unsigned char reversed_byte_table[256] = {0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA, 0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE, 0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9, 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD, 0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB, 0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF};

  public:

    template <unsigned char x, unsigned char LOG_N>
    inline static constexpr unsigned_t  reverse_byte( std::enable_if_t<LOG_N <= 8, bool> = true ) {
        return reversed_byte_table[x] >> (sizeof(unsigned char) * 8 - LOG_N);
    }

    template <unsigned short x, unsigned char LOG_N>
    inline static constexpr unsigned_t reverse_byte( std::enable_if_t<8 < LOG_N && LOG_N <= 16, bool> = true ) {
        constexpr unsigned short inByte0 = (x & 0xFF);
        constexpr unsigned short inByte1 = (x & 0xFF00) >> 8;
        return ((reversed_byte_table[inByte0] << 8) | reversed_byte_table[inByte1]) >> (sizeof(unsigned short) * 8 - LOG_N);
    }

    template <unsigned int x, unsigned char LOG_N>
    inline static constexpr unsigned_t reverse_byte( std::enable_if_t<16 < LOG_N && LOG_N <= 32, bool> = true ){
        constexpr unsigned int inByte0 = (x & 0xFF);
        constexpr unsigned int inByte1 = (x & 0xFF00) >> 8;
        constexpr unsigned int inByte2 = (x & 0xFF0000) >> 16;
        constexpr unsigned int inByte3 = (x & 0xFF000000) >> 24;
        return ((reversed_byte_table[inByte0] << 24) | (reversed_byte_table[inByte1] << 16) | (reversed_byte_table[inByte2] << 8) | reversed_byte_table[inByte3]) >> (sizeof(unsigned int) * 8 - LOG_N);
    }
};

/*
    Compile time generation of butterfly permutation into const array with variadic templates
*/

template <unsigned_t... args> struct ArrayHolder {
    static const unsigned_t data[sizeof...(args)];
};

template <unsigned_t... args>
const unsigned_t ArrayHolder<args...>::data[sizeof...(args)] = { args... };

template <unsigned_t N, unsigned char LOG_N, unsigned_t... args>
struct generate_permutation_impl {
    typedef typename generate_permutation_impl<N-1, LOG_N, BitReversal::reverse_byte<N, LOG_N>(), args...>::result result;
};

template <unsigned char LOG_N, unsigned_t... args>
struct generate_permutation_impl<0, LOG_N, args...> {
    typedef ArrayHolder<BitReversal::reverse_byte<0, LOG_N>(), args...> result;
};

template<unsigned char LOG_N>
struct generate_permutation {
    typedef typename generate_permutation_impl<(1 << LOG_N)-1, LOG_N>::result result;
};

/*
    Bit Reversal algorithm, needs compile time evaluable log_2 of table side size (b) and problem size (n)
*/

template <typename T, unsigned_t LOG_B, unsigned_t LOG_N>
class BitRev {
  public:

    constexpr static unsigned_t B = 1 << LOG_B;
    constexpr static unsigned_t B_4 = B >> 2;
    constexpr static unsigned_t B_2 = B >> 1;
    constexpr static unsigned_t B3_4 = B_4 + B_2;
    constexpr static unsigned_t LOG_B_BIT = (1 << LOG_B) - 1;

    constexpr static signed_t LOG_BR = LOG_N - 3 * LOG_B;
    constexpr static unsigned_t R = 1 << (LOG_N - 2 * LOG_B);
    constexpr static unsigned_t BR = B * R * 2;
    constexpr static unsigned_t PAR = 1 << ((LOG_N - 2 * LOG_B + 1) >> 1);

    inline static unsigned_t size() {
        return ((B_4 * (B_4 + 1) / 2 * 10) + (B_4 * (B_4 - 1) / 2 * 6)) * 2;
    }

    inline static T * allocate() {
        return (T*) malloc(sizeof(T) * size());
    }

    template <signed_t LBR>
    inline static unsigned_t rev_r( const unsigned_t & r, const unsigned_t * rev, std::enable_if_t<0 < LBR, bool> = true ) {

        const unsigned_t a = r & LOG_B_BIT;
        const unsigned_t b = (r - a) >> (unsigned_t) LOG_BR;
        return (rev[a] << (unsigned_t) LOG_BR) + rev[b];
    }

    template <signed_t LBR>
    inline static unsigned_t rev_r( const unsigned_t & r, const unsigned_t * rev, std::enable_if_t<0 == LBR, bool> = true ) {

        return rev[r];
    }

    template <signed_t LBR>
    inline static unsigned_t rev_r( const unsigned_t & r, const unsigned_t * rev, std::enable_if_t<LBR < 0, bool> = true ) {

        return rev[r] >> (unsigned_t) -LOG_BR;
    }

    inline static void reverse( T * __restrict const arr, unsigned_t nth ) {

        static_assert(2 * LOG_B <= LOG_N);      // can't use tables that wont fit LOG_N - if this fails, try lower LOG_B
        static_assert(LOG_N <= 4 * LOG_B);      // algorithm can effectively use LOG_B permutations only up to 4 times LOG_B problem size - if this fails, try higher LOG_B

        typedef typename generate_permutation<LOG_B>::result brev;

        T * __restrict array_block  = arr;
        T * __restrict array_block_ = arr;

        static T * __restrict cache_block;
        #pragma omp threadprivate(cache_block)

        #pragma omp parallel num_threads(nth)
        {
            cache_block = allocate();
        }

        //#pragma omp parallel
        for ( unsigned_t t = 0; t < R; t += PAR ) {
            #pragma omp parallel for num_threads(nth) schedule(static,1024)
            for ( unsigned_t p = 0; p < PAR; ++p ) {

                const unsigned_t r  = t + p;
                const unsigned_t r_ = rev_r<LOG_BR>(r, brev::data);
                const unsigned_t spec = r < r_ ? 0 : 1;
                array_block  = arr + r  * B * 2;
                array_block_ = arr + r * B * 2;

                unsigned_t block, k;
                /* LOAD */
                T * c = cache_block;
                for ( k = 0, block = spec; k < B_4; ++k ) {
                    T * a_b = array_block + k * BR;
                    for ( unsigned_t b = 0; b < block; ++b ) {          // Type 0111
                        T * a = a_b + brev::data[b] * 2;

                        c[0] = a[2];
                        c[1] = a[3];
                        c[2] = a[4];
                        c[3] = a[5];
                        c[4] = a[6];
                        c[5] = a[7];
                        c += 12;
                    }
                    for ( unsigned_t b = block++; b < B_4; ++b ) {      // Type 1111
                        T * a = a_b + brev::data[b] * 2;

                        c[0] = a[0];
                        c[1] = a[1];
                        c[2] = a[2];
                        c[3] = a[3];
                        c[4] = a[4];
                        c[5] = a[5];
                        c[6] = a[6];
                        c[7] = a[7];
                        c += 20;
                    }
                }

                for ( c = cache_block, block = spec; k < B_2; ++k ) {
                    T * a_b = array_block + k * BR;
                    for ( unsigned_t b = 0; b < block; ++b ) {          // Type 0101
                        T * a = a_b + brev::data[b] * 2;

                        c[10] = a[2];
                        c[11] = a[3];
                        c[8]  = a[6];
                        c[9]  = a[7];
                        c += 12;
                    }
                    for ( unsigned_t b = block++; b < B_4; ++b ) {      // Type 0111
                        T * a = a_b + brev::data[b] * 2;

                        c[16] = a[2];
                        c[17] = a[3];
                        c[18] = a[4];
                        c[19] = a[5];
                        c[10] = a[6];
                        c[11] = a[7];
                        c += 20;
                    }
                }

                for ( c = cache_block, block = spec; k < B3_4; ++k ) {
                    T * a_b = array_block + k * BR;
                    for ( unsigned_t b = 0; b < block; ++b ) {          // Type 0001
                        T * a = a_b + brev::data[b] * 2;

                        c[6] = a[6];
                        c[7] = a[7];
                        c += 12;
                    }
                    for ( unsigned_t b = block++; b < B_4; ++b ) {      // Type 0101
                        T * a = a_b + brev::data[b] * 2;

                        c[14] = a[2];
                        c[15] = a[3];
                        c[8]  = a[6];
                        c[9]  = a[7];
                        c += 20;
                    }
                }

                for ( c = cache_block, block = spec; k < B; ++k ) {
                    T * a_b = array_block + k * BR;
                    c += 12 * block;									// Type 0000
                    for ( unsigned_t b = block++; b < B_4; ++b ) {      // Type 0001
                        T * a = a_b + brev::data[b] * 2;

                        c[12] = a[6];
                        c[13] = a[7];
                        c += 20;
                    }
                }

                /* SWAP */
                const unsigned_t rspec = B_4 - spec;
                unsigned_t blockh, blockv;

                for ( k = B_4, blockv = rspec; k > 0; ) {
                    T * a_b = array_block_ + (B3_4 + --k) * BR;
                    c = cache_block;
                    blockh = rspec;
                    for ( unsigned_t b = 0; b < blockv; ++b ) {          // Type 1111
                        T * a = a_b + brev::data[b] * 2;
                        const unsigned_t off = k * 12 + (k > (b + spec) ? k - (b + spec) : 0) * 8;

                        std::swap(c[off + 6],  a[0]);
                        std::swap(c[off + 7],  a[1]);
                        std::swap(c[off + 8],  a[2]);
                        std::swap(c[off + 9],  a[3]);
                        std::swap(c[off + 10], a[4]);
                        std::swap(c[off + 11], a[5]);
                        std::swap(c[off + 12], a[6]);
                        std::swap(c[off + 13], a[7]);
                        c += B_4 * 12 + blockh-- * 8;
                    }
                    for ( unsigned_t b = blockv--; b < B_4; ++b ) {      // Type 1110
                        T * a = a_b + brev::data[b] * 2;
                        const unsigned_t off = k * 12 + (k > (b + spec) ? k - (b + spec) : 0) * 8;

                        std::swap(c[off + 4], a[0]);
                        std::swap(c[off + 5], a[1]);
                        std::swap(c[off + 6], a[2]);
                        std::swap(c[off + 7], a[3]);
                        std::swap(c[off + 8], a[4]);
                        std::swap(c[off + 9], a[5]);
                        c += B_4 * 12 + blockh-- * 8;
                    }
                }

                for ( k = B_4, blockv = rspec; k > 0; ) {
                    T * a_b = array_block_ + (B_2 + --k) * BR;
                    c = cache_block;
                    blockh = rspec;
                    for ( unsigned_t b = 0; b < blockv; ++b ) {          // Type 1110
                        T * a = a_b + brev::data[b] * 2;
                        const unsigned_t off = k * 12 + (k > (b + spec) ? k - (b + spec) : 0) * 8;

                        std::swap(c[off + 2],  a[0]);
                        std::swap(c[off + 3],  a[1]);
                        std::swap(c[off + 14], a[2]);
                        std::swap(c[off + 15], a[3]);
                        std::swap(c[off + 16], a[4]);
                        std::swap(c[off + 17], a[5]);
                        c += B_4 * 12 + blockh-- * 8;
                    }
                    for ( unsigned_t b = blockv--; b < B_4; ++b ) {      // Type 1010
                        T * a = a_b + brev::data[b] * 2;
                        const unsigned_t off = k * 12 + (k > (b + spec) ? k - (b + spec) : 0) * 8;

                        std::swap(c[off], a[0]);
                        std::swap(c[off + 1], a[1]);
                        std::swap(c[off + 10], a[4]);
                        std::swap(c[off + 11], a[5]);
                        c += B_4 * 12 + blockh-- * 8;
                    }
                }

                for ( k = B_4, blockv = rspec; k > 0; ) {
                    T * a_b = array_block_ + (B_4 + --k) * BR;
                    c = cache_block;
                    blockh = rspec;
                    for ( unsigned_t b = 0; b < blockv; ++b ) {          // Type 1010
                        T * a = a_b + brev::data[b] * 2;
                        const unsigned_t off = k * 12 + (k > (b + spec) ? k - (b + spec) : 0) * 8;

                        std::swap(c[off + 4], a[0]);
                        std::swap(c[off + 5], a[1]);
                        std::swap(c[off + 18], a[4]);
                        std::swap(c[off + 19], a[5]);
                        c += B_4 * 12 + blockh-- * 8;
                    }
                    for ( unsigned_t b = blockv--; b < B_4; ++b ) {      // Type 1000
                        T * a = a_b + brev::data[b] * 2;
                        const unsigned_t off = k * 12 + (k > (b + spec) ? k - (b + spec) : 0) * 8;

                        std::swap(c[off + 2], a[0]);
                        std::swap(c[off + 3], a[1]);
                        c += B_4 * 12 + blockh-- * 8;
                    }
                }

                for ( k = B_4, blockv = rspec; k > 0; ) {
                    T * a_b = array_block_ + --k * BR;
                    c = cache_block;
                    blockh = rspec;
                    for ( unsigned_t b = 0; b < blockv; ++b ) {          // Type 1000
                        T * a = a_b + brev::data[b] * 2;
                        const unsigned_t off = k * 12 + (k > (b + spec) ? k - (b + spec) : 0) * 8;

                        std::swap(c[off], a[0]);
                        std::swap(c[off + 1], a[1]);
                        c += B_4 * 12 + blockh-- * 8;
                    }
                    for ( unsigned_t b = blockv--; b < B_4; ++b ) {      // Type 0000
                        c += B_4 * 12 + blockh-- * 8;
                    }
                }

                /* STORE */
                for ( c = cache_block, k = 0, block = spec; k < B_4; ++k ) {
                    T * a_b = array_block + k * BR;
                    for ( unsigned_t b = 0; b < block; ++b ) {          // Type 0111
                        T * a = a_b + brev::data[b] * 2;
                        a[2] = c[0];
                        a[3] = c[1];
                        a[4] = c[2];
                        a[5] = c[3];
                        a[6] = c[4];
                        a[7] = c[5];
                        c += 12;
                    }
                    for ( unsigned_t b = block++; b < B_4; ++b ) {      // Type 1111
                        T * a = a_b + brev::data[b] * 2;
                        a[0] = c[0];
                        a[1] = c[1];
                        a[2] = c[2];
                        a[3] = c[3];
                        a[4] = c[4];
                        a[5] = c[5];
                        a[6] = c[6];
                        a[7] = c[7];
                        c += 20;
                    }
                }

                for ( c = cache_block, block = spec; k < B_2; ++k ) {
                    T * a_b = array_block + k * BR;
                    for ( unsigned_t b = 0; b < block; ++b ) {          // Type 0101
                        T * a = a_b + brev::data[b] * 2;
                        a[2] = c[10];
                        a[3] = c[11];
                        a[6] = c[8];
                        a[7] = c[9];
                        c += 12;
                    }
                    for ( unsigned_t b = block++; b < B_4; ++b ) {      // Type 0111
                        T * a = a_b + brev::data[b] * 2;
                        a[2] = c[16];
                        a[3] = c[17];
                        a[4] = c[18];
                        a[5] = c[19];
                        a[6] = c[10];
                        a[7] = c[11];
                        c += 20;
                    }
                }

                for ( c = cache_block, block = spec; k < B3_4; ++k ) {
                    T * a_b = array_block + k * BR;
                    for ( unsigned_t b = 0; b < block; ++b ) {          // Type 0001
                        T * a = a_b + brev::data[b] * 2;
                        a[6] = c[6];
                        a[7] = c[7];
                        c += 12;
                    }
                    for ( unsigned_t b = block++; b < B_4; ++b ) {      // Type 0101
                        T * a = a_b + brev::data[b] * 2;
                        a[2] = c[14];
                        a[3] = c[15];
                        a[6] = c[8];
                        a[7] = c[9];
                        c += 20;
                    }
                }

                for ( c = cache_block, block = spec; k < B; ++k ) {
                    T * a_b = array_block + k * BR;
                    c += 12 * block;									// Type 0000
                    for ( unsigned_t b = block++; b < B_4; ++b ) {      // Type 0001
                        T * a = a_b + brev::data[b] * 2;
                        a[6] = c[12];
                        a[7] = c[13];
                        c += 20;
                    }
                }
            }
        }
        #pragma omp parallel num_threads(nth)
        {
            free(cache_block);
        }
    }
};
