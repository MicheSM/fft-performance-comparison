#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <limits>
#include "config.h"
#include <arm_sve.h>
using std::cout, std::endl;
typedef uint64_t u64;
typedef double f64;

const f64 pi = M_PI;

void init_roots(f64* __restrict__ cosines, f64* __restrict__ sines, u64 n){
	for(u64 i = 0; i < n/2; ++i) sines[i] = sin(-2*pi*(f64)i/n);
	if(n == 2){
		cosines[0] = 1;
		cosines[1] = 0;
	}
	else {
		for(u64 i = 0; i < n/4; ++i) cosines[i] = -sines[i+n/4];
		for(u64 i = n/4; i < n/2; ++i) cosines[i] = sines[i-n/4];
	}
}

void fft_stockham(f64* __restrict__ re, f64* __restrict__ im, 
                  f64* __restrict__ re_tmp, f64* __restrict__ im_tmp,
                  f64* __restrict__ cosines, f64* __restrict__ sines, u64 n){
	assert((n & (n-1)) == 0 && n > 0);
	u64 logn = 63 - __builtin_clzll(n);
	u64 vector_step = svcntd();

	// Pointers for ping-pong buffering
	f64 *re_in = re, *im_in = im;
	f64 *re_out = re_tmp, *im_out = im_tmp;
	
	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		u64 num_groups = n / p;
		
		for(u64 group = 0; group < num_groups; ++group){
			for(u64 j = 0; j < halfp; j += vector_step){
				svbool_t predicate = svwhilelt_b64(j, halfp);
				
				// Calculate input indices for Stockham pattern
				u64 base_in_0 = group * halfp + j;
				u64 base_in_1 = base_in_0 + n/2;
				
				// Calculate output indices
				u64 base_out_even = group * p + j;
				u64 base_out_odd = base_out_even + halfp;
				
				// Load twiddle factors using gather
				svuint64_t root_indices = svindex_u64(j << (logn-logp), 1 << (logn-logp));
				svfloat64_t wre = svld1_gather_index(predicate, cosines, root_indices);
				svfloat64_t wim = svld1_gather_index(predicate, sines, root_indices);
				
				// Load input values
				svfloat64_t re0 = svld1(predicate, &re_in[base_in_0]);
				svfloat64_t im0 = svld1(predicate, &im_in[base_in_0]);
				svfloat64_t re1 = svld1(predicate, &re_in[base_in_1]);
				svfloat64_t im1 = svld1(predicate, &im_in[base_in_1]);
				
				// Complex multiply with twiddle: z = w * val1
				svfloat64_t zre = svmul_x(predicate, wre, re1);
				zre = svmls_x(predicate, zre, wim, im1);
				svfloat64_t zim = svmul_x(predicate, wre, im1);
				zim = svmla_x(predicate, zim, wim, re1);
				
				// Butterfly and store
				svst1(predicate, &re_out[base_out_even], svadd_x(predicate, re0, zre));
				svst1(predicate, &im_out[base_out_even], svadd_x(predicate, im0, zim));
				svst1(predicate, &re_out[base_out_odd], svsub_x(predicate, re0, zre));
				svst1(predicate, &im_out[base_out_odd], svsub_x(predicate, im0, zim));
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
	
	double minimum_time = std::numeric_limits<double>::max();
	for(int r = 0; r < 8; r++){
		auto startTime = high_resolution_clock::now();

		fft_stockham(re, im, re_tmp, im_tmp, cosines, sines, n);

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
	delete[] re_tmp;
	delete[] im_tmp;
	delete[] cosines;
	delete[] sines;
}
