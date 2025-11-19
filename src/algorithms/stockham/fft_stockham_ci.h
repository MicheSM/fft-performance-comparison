
#pragma once

#include <cstdint>

namespace stockham {

using u64 = uint64_t;
using f64 = double;

// Simple complex number struct used by the CI implementations.
struct cf64 {
	f64 re;
	f64 im;
};

// Initialize roots table (array of cf64 of length n/2).
void init_roots(cf64* roots, u64 n);

// Stockham FFT using contiguous cf64 arrays (complex struct, non-interleaved
// in memory as pair of doubles). `wave_tmp` must be a buffer of size `n`.
void fft_stockham(cf64* __restrict__ wave, cf64* __restrict__ wave_tmp,
				  cf64* __restrict__ roots, u64 n);

} // namespace stockham
