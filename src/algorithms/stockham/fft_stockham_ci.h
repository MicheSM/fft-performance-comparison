#include <cstdint>

namespace stockham {

using u64 = uint64_t;
using f64 = double;

struct cf64 {
	f64 re;
	f64 im;
};

void init_roots(cf64* roots, u64 n);

void fft_stockham(cf64* __restrict__ wave, cf64* __restrict__ wave_tmp,
				  cf64* __restrict__ roots, u64 n);

}
