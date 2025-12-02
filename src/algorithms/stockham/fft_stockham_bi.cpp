#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <chrono>
#include "config.h"
#include <limits>
using std::cout, std::endl;
typedef uint64_t u64;
typedef double f64;

const f64 pi = M_PI;

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

void fft_stockham(f64* __restrict__ re, f64* __restrict__ im, 
                  f64* __restrict__ re_tmp, f64* __restrict__ im_tmp,
                  f64* __restrict__ cosines, f64* __restrict__ sines, u64 n){
	assert((n & (n-1)) == 0 && n > 0); // n deve essere una potenza di due
	u64 logn = 63 - __builtin_clzll(n);
	
	// Pointers for ping-pong buffering
	f64 *re_in = re, *im_in = im;
	f64 *re_out = re_tmp, *im_out = im_tmp;
	
	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		u64 num_groups = n / p;
		
		for(u64 group = 0; group < num_groups; ++group){
			for(u64 j = 0; j < halfp; ++j){
				// Stockham indexing pattern
				u64 idx_in_0 = group * halfp + j;  // First half of input
				u64 idx_in_1 = idx_in_0 + n/2;     // Second half of input
				u64 idx_out_even = group * p + j;  // Even output
				u64 idx_out_odd = idx_out_even + halfp; // Odd output
				
				// Twiddle factor
				f64 wre = cosines[j << (logn-logp)];
				f64 wim = sines[j << (logn-logp)];
				
				// Load input values
				f64 re0 = re_in[idx_in_0];
				f64 im0 = im_in[idx_in_0];
				f64 re1 = re_in[idx_in_1];
				f64 im1 = im_in[idx_in_1];
				
				// Complex multiply with twiddle
				f64 zre = wre * re1 - wim * im1;
				f64 zim = wre * im1 + wim * re1;
				
				// Butterfly computation
				re_out[idx_out_even] = re0 + zre;
				im_out[idx_out_even] = im0 + zim;
				re_out[idx_out_odd] = re0 - zre;
				im_out[idx_out_odd] = im0 - zim;
			}
		}
		
		// Swap buffers for next iteration
		std::swap(re_in, re_out);
		std::swap(im_in, im_out);
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
	f64* re_tmp = new f64[n];
	f64* im_tmp = new f64[n];
	
	for(u64 i = 0; i < n; ++i){
		re[i] = 0;
		im[i] = 0;
	}
	
	f64* cosines = new f64[n/2];
	f64* sines = new f64[n/2];
	init_roots(cosines, sines, n);
	
	using std::chrono::high_resolution_clock;
	using std::chrono::time_point;
	
	std::vector<double> times;
	times.reserve(100);

	for (int r = 0; r < 32; r++) {
		auto startTime = high_resolution_clock::now();

		fft_stockham(re, im, re_tmp, im_tmp, cosines, sines, n);

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
	for(u64 i = 0; i < n; ++i) std::cout << re[i] << " " << im[i] << "\n";
    #endif
	
	delete[] re;
	delete[] im;
	delete[] re_tmp;
	delete[] im_tmp;
	delete[] cosines;
	delete[] sines;
}

