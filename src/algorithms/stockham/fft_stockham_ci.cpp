#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <complex>
#include "timer.h"
#include "config.h"
using std::cout, std::endl;
typedef uint64_t u64;
typedef double f64;
struct cf64{
	f64 re;
	f64 im;
};

const f64 pi = M_PI;

inline cf64 operator+(const cf64& a, const cf64& b){
	return {a.re+b.re, a.im + b.im};
}
inline cf64 operator-(const cf64& a, const cf64& b){
	return {a.re-b.re, a.im-b.im};
}
inline cf64 operator*(const cf64& a, const cf64& b){
	return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}

void init_roots(cf64* roots, u64 n){
	for(u64 i = 0; i < n/2; ++i) roots[i].im = sin(-2*pi*(f64)i/n);
	if(n==2){ // non posso sfruttare cos(x) = sin(x + pi/2) perchè calcolo sono in 0 e pi
		roots[0].re = 1;
		roots[1].re = 0;
	}
	else { // i segni sono al contrario perche gli angoli sono negativi
		for(u64 i = 0; i < n/4; ++i) roots[i].re = -roots[i+n/4].im;
		for(u64 i = n/4; i < n/2; ++i) roots[i].re = roots[i-n/4].im;
	}
}

void fft_stockham(cf64* __restrict__ wave, cf64* __restrict__ wave_tmp,
		  cf64* __restrict__ roots, u64 n){
	assert((n & (n-1)) == 0 && n > 0); // n deve essere una potenza di due
	u64 logn = 63 - __builtin_clzll(n);


	cf64 *wave_in = wave;
	cf64 *wave_out = wave_tmp;

	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		u64 num_groups = n / p;

		for(u64 group = 0; group < num_groups; ++group){
#ifdef PRAGMA_VECTOR
			#pragma GCC ivdep
#endif
			for(u64 j = 0; j < halfp; ++j){
				// stockham indexing pattern
				u64 idx_in_0 = group * halfp + j; // prima metà dell'input
				u64 idx_in_1 = idx_in_0 + n/2;    // seconda metà dell'input
				u64 idx_out_even = group * p + j; // output pari
				u64 idx_out_odd = idx_out_even + halfp; // output dispari
			
				// Twiddle factor and butterfly
				cf64 w = roots[j <<(logn-logp)];
				cf64 val0 = wave_in[idx_in_0];
				cf64 val1 = wave_in[idx_in_1];
				cf64 z = w * val1;

				wave_out[idx_out_even] = val0 + z;
				wave_out[idx_out_odd] = val0 - z;
			}	
		}

		std::swap(wave_in, wave_out);
	}
}

int main(int argc, char const * argv[]){
	if(argc != 2){
		std::cout << "expected 1 argument: n\n";
		exit(1);
	}
	u64 n = atoi(argv[1]);
	cf64* wave = new cf64[n];
	cf64* wave_tmp = new cf64[n];

	for(u64 i = 0; i < n; ++i){
		wave[i] = {0.4269 * cos(2*pi*(f64)i/n) + cos(2*pi*3*(f64)i/n),
			   0.4269 * sin(2*pi*(f64)i/n) + sin(2*pi*3*(f64)i/n)};
	}

	cf64* roots = new cf64[n/2];
	init_roots(roots,n);
	using std::chrono::high_resolution_clock;
	using std::chrono::time_point;
	auto startTime = high_resolution_clock::now();
	fft_stockham(wave, wave_tmp, roots, n);
	auto endTime = high_resolution_clock::now();
	double elapsed = std::chrono::duration<double, std::nano>(endTime - startTime).count();
	cout << n << " " << elapsed << endl; 

#ifndef DONTPRINT
	for(u64 i = 0; i < n; ++i) std::cout << wave[i].re << " " << wave[i].im << "\n";
#endif

	delete[] wave;
	delete[] wave_tmp;
	delete[] roots;
	
}





