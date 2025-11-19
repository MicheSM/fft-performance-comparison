#include <cstdint>

typedef uint64_t u64;
typedef double f64;

struct cf64 { f64 re; f64 im; };

void init_roots(cf64* roots, u64 n);
void fft(cf64* __restrict__ wave, cf64* __restrict__ roots, u64 n);