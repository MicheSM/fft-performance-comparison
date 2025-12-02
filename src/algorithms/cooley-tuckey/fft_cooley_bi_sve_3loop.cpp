#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <arm_sve.h>
#include <chrono>
#include <limits>
#include "config.h"
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

void fft(f64* __restrict__ re, f64* __restrict__ im, f64* __restrict__ cosines, f64* __restrict__ sines, u64 n){
	assert((n & (n-1)) == 0 && n > 0); // n deve essere una potenza di due
	u64 logn = 63 - __builtin_clzll(n);
	// bit reversal permutation con gather e scatter
	u64 vector_step = svcntd(); // quanti double in un vettore
	for(u64 i = 0; i < n; i += vector_step){
		svbool_t predicate = svwhilelt_b64(i, n); // vector_step elementi o fino in fondo
		svuint64_t indices = svindex_u64(i, 1); // vettore [i, i+vector_step)
		svuint64_t rev_indices = svrbit_u64_x(predicate, indices);
		rev_indices = svlsr_n_u64_m(predicate, rev_indices, 64-logn);
		svbool_t swaps = svcmpgt(predicate, indices, rev_indices);
		// carica i valori da scambiare
		svfloat64_t sv_re = svld1(swaps, &re[i]);
		svfloat64_t sv_rev_re = svld1_gather_index(swaps, re, rev_indices);
		// rimettili al contrario
		svst1_scatter_index(swaps, re, rev_indices, sv_re);
		svst1(swaps, &re[i], sv_rev_re);
		// stessa cosa parti immaginarie
		svfloat64_t sv_im = svld1(swaps, &im[i]);
		svfloat64_t sv_rev_im = svld1_gather_index(swaps, im, rev_indices);
		svst1_scatter_index(swaps, im, rev_indices, sv_im);
		svst1(swaps, &im[i], sv_rev_im);
	}

	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		for(u64 i = 0; i < n; i += p){
			for(u64 j = 0; j < halfp; j += vector_step){
				svbool_t predicate = svwhilelt_b64(j, halfp);
				// indici "pari" e "dispari" corrispondono alla versione ricorsiva di fft
				u64 even = i+j, odd = i+j+halfp;

				svuint64_t root_indices = svindex_u64(j << (logn-logp), 1 << (logn-logp));
				svfloat64_t wre = svld1_gather_index(predicate, cosines, root_indices);
				svfloat64_t wim = svld1_gather_index(predicate, sines, root_indices);

				svfloat64_t re_even = svld1(predicate, &re[even]);
				svfloat64_t re_odd = svld1(predicate, &re[odd]);
				svfloat64_t im_even = svld1(predicate, &im[even]);
				svfloat64_t im_odd = svld1(predicate, &im[odd]);

				// z = w * odd
				svfloat64_t zre = svmul_x(predicate, wre, re_odd);
				zre = svmls_x(predicate, zre, wim, im_odd); // zre - wim * im_odd
				svfloat64_t zim = svmul_x(predicate, wre, im_odd);
				zim = svmla_x(predicate, zim, wim, re_odd); // zim + wim * re_odd

				// odd := even - z = even - w * odd
				svst1(predicate, &re[odd], svsub_x(predicate, re_even, zre));
				svst1(predicate, &im[odd], svsub_x(predicate, im_even, zim));

				// even := even + z = even + w * odd
				svst1(predicate, &re[even], svadd_x(predicate, re_even, zre));
				svst1(predicate, &im[even], svadd_x(predicate, im_even, zim));
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
	for(int r = 0; r < 100; r++){
		auto startTime = high_resolution_clock::now();

		fft(re, im, cosines, sines, n);

		auto endTime = high_resolution_clock::now();

		double elapsed = std::chrono::duration<double, std::nano>(endTime - startTime).count();

		if (elapsed < minimum_time) {
        	minimum_time = elapsed;
    	}
	}
	cout << n << " " << minimum_time << endl; 
	
#ifndef DONTPRINT
	for(u64 i = 0; i < n; ++i) std::cout << re[i] << " " << im[i] << "\n";
#endif
	delete[] re;
	delete[] im;
	delete[] cosines;
	delete[] sines;
}
