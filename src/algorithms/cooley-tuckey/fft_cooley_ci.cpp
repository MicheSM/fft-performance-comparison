#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <complex>
#include <algorithm>
#include <chrono>
#include <limits>
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
	return {a.re+b.re, a.im+b.im};
}
inline cf64 operator-(const cf64& a, const cf64& b){
	return {a.re-b.re, a.im-b.im};
}
inline cf64 operator*(const cf64& a, const cf64& b){
	return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}

inline u64 reverse_bits(u64 x, u64 nbits){
#ifdef USE_RBIT_ASM
	__asm__("rbit %0, %1" : "=r"(x) : "r"(x));
#else
	x = ((x << 1) & 0xaaaaaaaaaaaaaaaa) | ((x & 0xaaaaaaaaaaaaaaaa) >> 1);
	x = ((x << 2) & 0xcccccccccccccccc) | ((x & 0xcccccccccccccccc) >> 2);
	x = ((x << 4) & 0xf0f0f0f0f0f0f0f0) | ((x & 0xf0f0f0f0f0f0f0f0) >> 4);
	x = ((x << 8) & 0xff00ff00ff00ff00) | ((x & 0xff00ff00ff00ff00) >> 8);
	x = ((x << 16) & 0xffff0000ffff0000) | ((x & 0xffff0000ffff0000) >> 16);
	x = (x << 32) | (x >> 32);
#endif
	return x >> (64 - nbits);
}

void init_roots(cf64* roots, u64 n){
	for(u64 i = 0; i < n/2; ++i) roots[i].im = sin(-2*pi*(f64)i/n);
	if(n == 2){ // non posso sfruttare cos(x) = sin(x + pi/2) perché calcolo solo in 0 e pi
		roots[0].re = 1;
		roots[1].re = 0;
	}
	else { // i segni sono al contrario perché gli angoli sono negativi
		for(u64 i = 0; i < n/4; ++i) roots[i].re = -roots[i+n/4].im;
		for(u64 i = n/4; i < n/2; ++i) roots[i].re = roots[i-n/4].im;
	}
}

void fft(cf64* __restrict__ wave, cf64* __restrict__ roots, u64 n){
	assert((n & (n-1)) == 0 && n > 0); // n deve essere una potenza di due
	u64 logn = 63 - __builtin_clzll(n);

	// bit reversal permutation con gli scambi
	for(u64 i = 0; i < n; ++i){
		u64 j = reverse_bits(i, logn);
		if(i < j){
			std::swap(wave[i], wave[j]);
		}
	}
	
	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		for(u64 i = 0; i < n; i += p){
#ifdef PRAGMA_VECTOR
			#pragma GCC ivdep
#endif
			for(u64 j = 0; j < halfp; ++j){
				u64 even = i+j, odd = i+j+halfp;
				cf64 w = roots[j << (logn-logp)];
				cf64 z = w * wave[odd];
				wave[odd] = wave[even] - z;
				wave[even] = wave[even] + z;
			}
		}
	}
}

int main(int argc, char const * argv[]){
	if(argc != 2){
		std::cout << "expected 1 argument: n\n";
		exit(1);
	}
	u64 n = atoi(argv[1]);
	cf64* wave = new cf64[n];
	for(u64 i = 0; i < n; ++i){
		wave[i] = {0, 0};
	}
	
	cf64* roots = new cf64[n/2];
	init_roots(roots, n);
	using std::chrono::high_resolution_clock;
	using std::chrono::time_point;
	
	std::vector<double> times;
	times.reserve(100);

	for (int r = 0; r < 32; r++) {
		auto startTime = high_resolution_clock::now();

		fft(wave, roots, n);

		auto endTime = high_resolution_clock::now();

		double elapsed = std::chrono::duration<double, std::nano>(endTime - startTime).count();
		times.push_back(elapsed);
	}

	// sort the measured times
	std::sort(times.begin(), times.end());
	
	// compute the median
	double median_time;
	int size = times.size();

	if (size % 2 == 0) {
		// even number of elements → average the two middle ones
		median_time = (times[size/2 - 1] + times[size/2]) / 2.0;
	} else {
		// odd number of elements → middle element
		median_time = times[size/2];
	}
	cout << n << " " << median_time << endl; 


	
#ifndef DONTPRINT
	for(u64 i = 0; i < n; ++i) std::cout << wave[i].re << " " << wave[i].im << "\n";
#endif
	delete[] wave;
	delete[] roots;
}
