#include "bitrev_complex_omp.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>

#define native_alignment 32

template <typename T, size_t n, size_t m, size_t b>
void time2n( uint32_t nth, uint32_t tests = 10 ) {

    using namespace std::chrono;

    std::vector<double> mflops(tests);
    std::vector<duration<double> > time(tests);

    T * input  = (T*)aligned_alloc(32, 2 * m * sizeof(T));

    BitRev<T, b, n> br;

    for ( uint32_t k = 0; k < tests; ++k ) {

        for ( uint32_t i = 0; i < m; ++i ) {
            input[i*2]   = (T)0.;
            input[i*2+1] = (T)0.;
        }

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        br.reverse(input, nth);

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

        time[k] = time_span;

        // 5 * log_2(n) / time

        mflops[k] = 5 * m * n / (std::log(2.0)*time_span.count() * 1000000.0F);

    }

    free(input);

    double mintime = time[0].count();
    double bestflops = mflops[0];

    for ( uint32_t k = 1; k < tests; ++k ) {
        if ( time[k].count() < mintime ) {
            mintime = time[k].count();
            bestflops = mflops[k];
        }
    }

    std::cout<<n<<'\t'<<bestflops<<'\t'<<std::fixed <<std::setprecision(9)<<(mintime * 1000.0F)<<std::endl;

}

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

#define N 24

int main() {
    for ( int t = 1; t < 13; t++ ) {
        omp_set_num_threads(t);
        #pragma omp parallel
        {
            #pragma omp master
            printf("  Using %d threads.\n", omp_get_num_threads());
        }
        std::cout<<"#size"<<'\t'<<"speed(mflops)"<<'\t'<<"time(ms)"<<std::endl;
        time2n<double, N, (1 << N), 6>(t);
    }

    double * arr = (double*)aligned_alloc(32, 2 * (1 << N) * sizeof(double));
    for ( int i = 0; i < (1 << N); ++i ) { arr[i*2] = arr[i*2+1] = i; }
    BitRev<double, 6, N> br;
    br.reverse(arr, 12);
    revbin_permute(arr, (1 << N));
    for ( int i = 0; i < std::min((1 << N), 256); ++i ) { std::cout << arr[i*2] <<  '\t'; if (i%8 == 7) std::cout <<std::endl;} std::cout << std::endl; //" " << arr[i*2+1] <<*/
    return 0;
}
