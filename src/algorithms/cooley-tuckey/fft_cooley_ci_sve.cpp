#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <complex>
#include <arm_sve.h>
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

	// bit reversal permutation con gather e scatter
	u64 vector_step = svcntd(); // quanti double in un vettore
	svuint64_t sv_0101 = svindex_u64(0, 1); // 01234567
	sv_0101 = svand_x(svptrue_b64(), sv_0101, 1); // 01010101
	// svld2/svst2 che è fatto per prendere coppie non fa gather e scatter
	// quindi consideriamoli come 2n singoli double, pari = reali, dispari = immaginari
	// vogliamo quindi fare rbit dell'indice ma il lsb lasciarlo lsb
	f64* wave_data = (f64*) wave;
	f64* roots_data = (f64*) roots;
	for(u64 i = 0; i < 2*n; i += vector_step){
		svbool_t predicate = svwhilelt_b64(i, 2*n); // vector_step elementi o fino in fondo
		svuint64_t indices = svindex_u64(i, 1); // vettore [i, i+vector_step)
		svuint64_t rev_indices = svlsr_m(predicate, indices, 1); // butto via il lsb, lo rimetto dopo aver reversato
		// resta uno zero dove prima c'era il msb, questo dopo rbit e lsr diventa lo spazio per rimettere il lsb
		rev_indices = svrbit_u64_x(predicate, rev_indices);
		rev_indices = svlsr_n_u64_m(predicate, rev_indices, 64-(logn+1));
		rev_indices = svorr_x(predicate, rev_indices, sv_0101);

		svbool_t swaps = svcmpgt(predicate, indices, rev_indices);
		// carica i valori da scambiare
		svfloat64_t sv_wave = svld1(swaps, wave_data + i);
		svfloat64_t sv_rev = svld1_gather_index(swaps, wave_data, rev_indices);
		// rimettili al contrario
		svst1_scatter_index(swaps, wave_data, rev_indices, sv_wave);
		svst1(swaps, wave_data + i, sv_rev);
	}

	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		for(u64 i = 0; i < n; i += vector_step/2){
			// se il bit halfp è settato, siamo nella metà "dispari", possiamo passare oltre
			// senza questo if, fai un paio di operazioni in più e viene il predicato tutto 0
			if(i & halfp) i += halfp;
			svbool_t buffer_pred = svwhilelt_b64(2*i, 2*n);

			svuint64_t indices = svindex_u64(2*i, 1);
			indices = svlsr_x(buffer_pred, indices, 1); // i i i+1 i+1 i+2 i+2 ...
			// devo mantenere attivi solo gli indici col bit halfp non settato
			// (la metà sinistra del dft che sto considerando, gli indici "pari")
			svuint64_t and_indices = svand_z(buffer_pred, indices, halfp);
			// a coppie a differenza di fft_sve.cpp
			svbool_t halfp_unset_pred = svcmpeq(buffer_pred, and_indices, 0);

			u64 even = i, odd = i + halfp;
			// root_indices = 2istep, 2istep+1, 2(i+1)step, 2(i+1)step+1, ... modulo 2p
			svuint64_t root_indices = indices; // ce l'abbiamo già i i i+1 i+1 j+2 j+2 ...
			root_indices = svand_x(halfp_unset_pred, root_indices, p-1); // tutti & (p-1) --> indici nel singolo dft che è grande p
			root_indices = svlsl_x(halfp_unset_pred, root_indices, logn-logp+1); // *= step angoli, *= 2
			root_indices = svorr_x(halfp_unset_pred, root_indices, sv_0101); // rendi dispari i dispari

			// fai copy paste da quando definisce w
			svfloat64_t w = svld1_gather_index(halfp_unset_pred, roots_data, root_indices);
			svfloat64_t wave_even = svld1(halfp_unset_pred, (f64*)(&wave[even]));
			svfloat64_t wave_odd = svld1(halfp_unset_pred, (f64*)(&wave[odd]));
			// z = w * odd
			svfloat64_t z = svdup_f64(0);
			z = svcmla_x(halfp_unset_pred, z, w, wave_odd, 0);
			z = svcmla_x(halfp_unset_pred, z, w, wave_odd, 90);
			// odd := even - z = even - w * odd; even := even + z = even + w * odd
			svst1(halfp_unset_pred, (f64*)&wave[odd], svsub_x(halfp_unset_pred, wave_even, z));
			svst1(halfp_unset_pred, (f64*)&wave[even], svadd_x(halfp_unset_pred, wave_even, z));
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
	times.reserve(8);

	for (int r = 0; r < 8; r++) {
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
	cout << n << " " << times.at(0) << endl; 

#ifndef DONTPRINT
	for(u64 i = 0; i < n; ++i) std::cout << wave[i].re << " " << wave[i].im << "\n";
#endif
	delete[] wave;
	delete[] roots;
}
