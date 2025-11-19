#include <cstdint>

typedef uint64_t u64;
typedef double f64;

void init_roots(f64* __restrict__ cosines, f64* __restrict__ sines, u64 n);
void fft(f64* __restrict__ re, f64* __restrict__ im, f64* __restrict__ cosines, f64* __restrict__ sines, u64 n);