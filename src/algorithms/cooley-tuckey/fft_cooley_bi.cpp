#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include "config.h"
#include <chrono>
#include <limits>
using std::cout, std::endl;
typedef uint64_t u64;
typedef double f64;

const f64 pi = M_PI;

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

void init_roots(f64* __restrict__ cosines, f64* __restrict__ sines, u64 n){
	for(u64 i = 0; i < n/2; ++i) sines[i] = sin(-2*pi*(f64)i/n);
	if(n == 2){ // non posso sfruttare cos(x) = sin(x + pi/2) perché calcolo solo in 0 e pi
		cosines[0] = 1;
		cosines[1] = 0;
	}
	else { // i segni sono al contrario perché gli angoli sono negativi
		for(u64 i = 0; i < n/4; ++i) cosines[i] = -sines[i+n/4];
		for(u64 i = n/4; i < n/2; ++i) cosines[i] = sines[i-n/4];
	}
}

void fft(f64* __restrict__ re, f64* __restrict__ im, f64* __restrict__ cosines, f64* __restrict__ sines, u64 n){
	assert((n & (n-1)) == 0 && n > 0); // n deve essere una potenza di due
	u64 logn = 63 - __builtin_clzll(n);

	// bit reversal permutation con gli scambi
	for(u64 i = 0; i < n; ++i){
		u64 j = reverse_bits(i, logn);
		if(i < j){
			std::swap(re[i], re[j]);
			std::swap(im[i], im[j]);
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
				// i+j = "even" index, i+j+halfp = "odd" index
				u64 even = i+j, odd = i+j+halfp;

				// one of the p-th roots of unity
				f64 wre = cosines[j << (logn-logp)];
				f64 wim =   sines[j << (logn-logp)];

				// z = w * odd
				f64 zre = wre * re[odd] - wim * im[odd];
				f64 zim = wre * im[odd] + wim * re[odd];

				// ""odd"" = even - z = even - w * odd
				re[odd] = re[even] - zre;
				im[odd] = im[even] - zim;

				// ""even"" = even + z = even + w * odd
				re[even] = re[even] + zre;
				im[even] = im[even] + zim;
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
	f64* re = new f64[n];
	f64* im = new f64[n];
	for(u64 i = 0; i < n; ++i){
		re[i] = 0;
		im[i] = 0;
	}
	
	f64* cosines = new f64[n/2];
	f64* sines = new f64[n/2];
	init_roots(cosines, sines, n);
	
	using std::chrono::high_resolution_clock;
	using std::chrono::time_point;
	
	double minimum_time = std::numeric_limits<double>::max();
	for(int r = 0; r < 8; r++){
		auto startTime = high_resolution_clock::now();

		fft(re, im, cosines, sines, n);

		auto endTime = high_resolution_clock::now();

		double elapsed = std::chrono::duration<double, std::nano>(endTime - startTime).count();

		if (elapsed < minimum_time) {
        	minimum_time = elapsed;
    	}
	}
	cout << n << " " << minimum_time; 
	
#ifndef DONTPRINT
	for(u64 i = 0; i < n; ++i) std::cout << re[i] << " " << im[i] << "\n";
#endif
	delete[] re;
	delete[] im;
	delete[] cosines;
	delete[] sines;
}
