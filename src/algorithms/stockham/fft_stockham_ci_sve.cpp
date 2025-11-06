#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <complex>
#include <arm_sve.h>
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
timer fft_timer;

inline cf64 operator+(const cf64& a, const cf64& b){
	return {a.re+b.re, a.im+b.im};
}
inline cf64 operator-(const cf64& a, const cf64& b){
	return {a.re-b.re, a.im-b.im};
}
inline cf64 operator*(const cf64& a, const cf64& b){
	return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}

void init_roots(cf64* roots, u64 n){
	for(u64 i = 0; i < n/2; ++i) roots[i].im = sin(-2*pi*(f64)i/n);
	if(n == 2){
		roots[0].re = 1;
		roots[1].re = 0;
	}
	else {
		for(u64 i = 0; i < n/4; ++i) roots[i].re = -roots[i+n/4].im;
		for(u64 i = n/4; i < n/2; ++i) roots[i].re = roots[i-n/4].im;
	}
}

void fft_stockham(cf64* __restrict__ wave, cf64* __restrict__ wave_tmp,
                  cf64* __restrict__ roots, u64 n){
	assert((n & (n-1)) == 0 && n > 0);
	u64 logn = 63 - __builtin_clzll(n);
	u64 vector_step = svcntd();
	
	// Setup for interleaved complex representation
	svuint64_t sv_0101 = svindex_u64(0, 1);
	sv_0101 = svand_x(svptrue_b64(), sv_0101, 1);
	f64* wave_data = (f64*) wave;
	f64* wave_tmp_data = (f64*) wave_tmp;
	f64* roots_data = (f64*) roots;
	
	// No bit reversal needed for Stockham!
	fft_timer.print_time("fft bit reversal done");
	
	// Pointers for ping-pong buffering
	f64 *data_in = wave_data;
	f64 *data_out = wave_tmp_data;
	
	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		u64 num_groups = n / p;
		
		for(u64 group = 0; group < num_groups; ++group){
			for(u64 j = 0; j < halfp; j += vector_step/2){
				svbool_t predicate = svwhilelt_b64(2*j, p);
				
				// Calculate input indices for Stockham pattern
				u64 base_in_0 = group * halfp + j;
				u64 base_in_1 = base_in_0 + n/2;
				
				// Calculate output indices
				u64 base_out_even = group * p + j;
				u64 base_out_odd = base_out_even + halfp;
				
				// Load twiddle factors
				svuint64_t root_indices = svindex_u64(2*j, 1);
				root_indices = svlsr_x(predicate, root_indices, 1);
				root_indices = svlsl_x(predicate, root_indices, logn-logp+1);
				root_indices = svorr_x(predicate, root_indices, sv_0101);
				svfloat64_t w = svld1_gather_index(predicate, roots_data, root_indices);
				
				// Load input values (complex interleaved)
				svfloat64_t val0 = svld1(predicate, &data_in[2*base_in_0]);
				svfloat64_t val1 = svld1(predicate, &data_in[2*base_in_1]);
				
				// Complex multiply: z = w * val1
				svfloat64_t z = svdup_f64(0);
				z = svcmla_x(predicate, z, w, val1, 0);
				z = svcmla_x(predicate, z, w, val1, 90);
				
				// Butterfly and store
				svst1(predicate, &data_out[2*base_out_even], svadd_x(predicate, val0, z));
				svst1(predicate, &data_out[2*base_out_odd], svsub_x(predicate, val0, z));
			}
		}
		
		// Swap buffers for next iteration
		std::swap(data_in, data_out);
	}
	
	// If odd number of stages, copy result back to original array
	if(logn & 1){
		for(u64 i = 0; i < 2*n; i += vector_step){
			svbool_t predicate = svwhilelt_b64(i, 2*n);
			svfloat64_t vals = svld1(predicate, &wave_tmp_data[i]);
			svst1(predicate, &wave_data[i], vals);
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
	cf64* wave_tmp = new cf64[n];
	
	for(u64 i = 0; i < n; ++i){
		wave[i] = {0.4269 * cos(2*pi*(f64)i/n) + cos(2*pi*3*(f64)i/n), 
		           0.4269 * sin(2*pi*(f64)i/n) + sin(2*pi*3*(f64)i/n)};
	}
	fft_timer.print_time("generated input");
	
	cf64* roots = new cf64[n/2];
	init_roots(roots, n);
	fft_timer.print_time("initialized vector of roots, starting fft");
	
	fft_stockham(wave, wave_tmp, roots, n);
	fft_timer.print_time("fft done");
	
#ifndef DONTPRINT
	for(u64 i = 0; i < n; ++i) std::cout << wave[i].re << " " << wave[i].im << "\n";
#endif
	
	delete[] wave;
	delete[] wave_tmp;
	delete[] roots;
}
