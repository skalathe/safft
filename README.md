All FFT files are benchmarks combined with tested algorithm actually practicing forward FFT, backward(inverse) will be supported once forward algorithms are finetuned.

fft(4/8)avx(512)_recit.cpp are radix-(4/8) FFT algorithms, 512 stands for avx512 support otherwise avx is supported

fft(4/8)avx(512)_recitbr.cpp is experimental version with better approach for bit reversal known as COBRA algorithm, it is actually WIP since this algorithm has to be pruned and vectorized
