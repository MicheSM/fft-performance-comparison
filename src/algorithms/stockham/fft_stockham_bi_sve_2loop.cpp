#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <arm_sve.h>
#include "timer.h"
#include "config.h"
using std::cout, std::endl;
typedef uint64_t u64;
typedef double f64;

const f64 pi = M_PI;
timer fft_timer;

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
	
	// No bit reversal needed for Stockham!
	fft_timer.print_time("fft bit reversal done");
	
	// Pointers for ping-pong buffering
	f64 *re_in = re, *im_in = im;
	f64 *re_out = re_tmp, *im_out = im_tmp;
	
	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		
		// MERGED LOOP: iterate over all butterfly positions
		for(u64 j = 0; j < n/2; j += vector_step){
			// Skip if we're in the "odd" half of a group
			// (j / halfp) & 1 tells us if we're in odd or even group half
			if((j / halfp) & 1) {
				j += halfp;
				// Need to recheck bounds after skip
				if(j >= n/2) break;
			}
			
			svbool_t predicate = svwhilelt_b64(j, n/2);
			
			// Create indices for this vector
			svuint64_t indices = svindex_u64(j, 1);
			
			// Filter to only elements in "even" half of their group
			// For position j, we want (j % p) < halfp
			svuint64_t position_in_group = svand_z(predicate, indices, p-1);
			svbool_t active_pred = svcmplt(predicate, position_in_group, halfp);
			
			// Compute group number for each element
			svuint64_t group_num = svlsr_x(active_pred, indices, logp-1);
			
			// Calculate input indices for Stockham pattern
			// base_in_0 = group * halfp + (j % halfp)
			svuint64_t j_in_group = svand_x(active_pred, indices, halfp-1);
			svuint64_t base_in_0 = svmul_n_u64_x(active_pred, group_num, halfp);
			base_in_0 = svadd_x(active_pred, base_in_0, j_in_group);
			svuint64_t base_in_1 = svadd_n_u64_x(active_pred, base_in_0, n/2);
			
			// Calculate output indices
			// base_out_even = group * p + (j % halfp)
			svuint64_t base_out_even = svmul_n_u64_x(active_pred, group_num, p);
			base_out_even = svadd_x(active_pred, base_out_even, j_in_group);
			svuint64_t base_out_odd = svadd_n_u64_x(active_pred, base_out_even, halfp);
			
			// Load twiddle factors
			svuint64_t root_indices = svlsl_x(active_pred, j_in_group, logn-logp);
			svfloat64_t wre = svld1_gather_index(active_pred, cosines, root_indices);
			svfloat64_t wim = svld1_gather_index(active_pred, sines, root_indices);
			
			// Load input values using gather
			svfloat64_t re0 = svld1_gather_index(active_pred, re_in, base_in_0);
			svfloat64_t im0 = svld1_gather_index(active_pred, im_in, base_in_0);
			svfloat64_t re1 = svld1_gather_index(active_pred, re_in, base_in_1);
			svfloat64_t im1 = svld1_gather_index(active_pred, im_in, base_in_1);
			
			// Complex multiply with twiddle: z = w * val1
			svfloat64_t zre = svmul_x(active_pred, wre, re1);
			zre = svmls_x(active_pred, zre, wim, im1);
			svfloat64_t zim = svmul_x(active_pred, wre, im1);
			zim = svmla_x(active_pred, zim, wim, re1);
			
			// Butterfly and store using scatter
			svst1_scatter_index(active_pred, re_out, base_out_even, 
			                    svadd_x(active_pred, re0, zre));
			svst1_scatter_index(active_pred, im_out, base_out_even, 
			                    svadd_x(active_pred, im0, zim));
			svst1_scatter_index(active_pred, re_out, base_out_odd, 
			                    svsub_x(active_pred, re0, zre));
			svst1_scatter_index(active_pred, im_out, base_out_odd, 
			                    svsub_x(active_pred, im0, zim));
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
		re[i] = 0.4269 * cos(2*pi*(f64)i/n) + cos(2*pi*3*(f64)i/n);
		im[i] = 0.4269 * sin(2*pi*(f64)i/n) + sin(2*pi*3*(f64)i/n);
	}
	fft_timer.print_time("generated input");
	
	f64* cosines = new f64[n/2];
	f64* sines = new f64[n/2];
	init_roots(cosines, sines, n);
	fft_timer.print_time("initialized vector of roots, starting fft");
	
	fft_stockham(re, im, re_tmp, im_tmp, cosines, sines, n);
	fft_timer.print_time("fft done");
	
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
