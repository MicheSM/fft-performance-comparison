#include <cstdint>

using u64 = uint64_t;
using f64 = double;

void init_roots(f64* __restrict__ cosines, f64* __restrict__ sines, u64 n);

void fft_stockham(f64* __restrict__ re, f64* __restrict__ im,
				  f64* __restrict__ re_tmp, f64* __restrict__ im_tmp,
				  f64* __restrict__ cosines, f64* __restrict__ sines,
				  u64 n);
