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
		for(u64 i = 0; i < n; i += vector_step){
			// se il bit halfp è settato, siamo nella metà "dispari", possiamo passare oltre
			// senza questo if, fai un paio di operazioni in più e viene il predicato tutto 0
			if(i & halfp) i += halfp;
			svbool_t buffer_pred = svwhilelt_b64(i, n);
			svuint64_t indices = svindex_u64(i, 1);

			// devo mantenere attivi solo gli indici col bit halfp non settato
			// (la metà sinistra del dft che sto considerando, gli indici "pari")
			svuint64_t and_indices = svand_z(buffer_pred, indices, halfp);
			svbool_t halfp_unset_pred = svcmpeq(buffer_pred, and_indices, 0);
			// if(!svptest_any(buffer_pred, halfp_unset_pred)) continue; // nulla da fare

			u64 even = i, odd = i + halfp;
			svuint64_t root_indices = svindex_u64(i, 1); // i i+1 i+2 ...
			root_indices = svand_x(halfp_unset_pred, root_indices, p-1); // tutti & (p-1) --> indici nel singolo dft che è grande p
			root_indices = svlsl_n_u64_m(halfp_unset_pred, root_indices, logn-logp); // left shift perché vogliamo j*step

			// copy paste brutale della versione vecchia dopo aver trovato gli indici
			// con halfp_unset_pred al posto di predicate
			svfloat64_t wre = svld1_gather_index(halfp_unset_pred, cosines, root_indices);
			svfloat64_t wim = svld1_gather_index(halfp_unset_pred, sines, root_indices);

			svfloat64_t re_even = svld1(halfp_unset_pred, &re[even]);
			svfloat64_t re_odd = svld1(halfp_unset_pred, &re[odd]);
			svfloat64_t im_even = svld1(halfp_unset_pred, &im[even]);
			svfloat64_t im_odd = svld1(halfp_unset_pred, &im[odd]);

			// z = w * odd
			svfloat64_t zre = svmul_x(halfp_unset_pred, wre, re_odd);
			zre = svmls_x(halfp_unset_pred, zre, wim, im_odd); // zre - wim * im_odd
			svfloat64_t zim = svmul_x(halfp_unset_pred, wre, im_odd);
			zim = svmla_x(halfp_unset_pred, zim, wim, re_odd); // zim + wim * re_odd

			// odd := even - z = even - w * odd
			svst1(halfp_unset_pred, &re[odd], svsub_x(halfp_unset_pred, re_even, zre));
			svst1(halfp_unset_pred, &im[odd], svsub_x(halfp_unset_pred, im_even, zim));

			// even := even + z = even + w * odd
			svst1(halfp_unset_pred, &re[even], svadd_x(halfp_unset_pred, re_even, zre));
			svst1(halfp_unset_pred, &im[even], svadd_x(halfp_unset_pred, im_even, zim));
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
