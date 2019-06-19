All FFT files are benchmarks combined with tested algorithm actually practicing forward FFT, backward(inverse) will be supported once forward algorithms are finetuned.

fft(4/8)avx(512)_recit.cpp are radix-(4/8) FFT algorithms, 512 stands for avx512 support otherwise avx is supported

fft(4/8)avx(512)_recitbr.cpp is experimental version with better approach for bit reversal known as COBRA algorithm

*19.06.2019*
fft(4/8)avx_recit.cpp now use new bit reversal bitrev_complex.hpp, it is compile time so fo max efficiency build only one plan structure for fft and bitrev, these structures are reusable for problems of same size.

bitrev_complex.cpp files are benchmarks with example of usage

bitrev omp version is still WIP and doesnt scale very well

fft4avx_recit_par.cpp omp version have bitrev commented and waits for completion of omp bitrev

