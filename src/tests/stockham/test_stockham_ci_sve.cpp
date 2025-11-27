#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <new>
#include <random>
#include <ratio>
#include <vector>
#include <arm_sve.h>

using std::complex;
using std::size_t;
using std::uint64_t;
using std::vector;
using std::cout;
using std::endl;
typedef uint64_t u64;
typedef double f64;

struct cf64{
	f64 re;
	f64 im;
};

inline cf64 operator+(const cf64& a, const cf64& b){
	return {a.re+b.re, a.im + b.im};
}
inline cf64 operator-(const cf64& a, const cf64& b){
	return {a.re-b.re, a.im-b.im};
}
inline cf64 operator*(const cf64& a, const cf64& b){
	return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}


// Private function prototypes
static vector<complex<double> > naiveDft(const complex<double> *input, size_t n);
void init_roots(cf64* roots, u64 n);
// Global variables
const f64 pi = M_PI;
static std::default_random_engine randGen((std::random_device())());
static std::uniform_real_distribution<double> realDist(-1, 1);

// Computes the discrete Fourier transform using the naive O(n^2) time algorithm.
static vector<complex<double> > naiveDft(const complex<double> *input, size_t n) {
	vector<complex<double> > output;
	for (size_t k = 0; k < n; k++) {  // For each output element
		complex<double> sum(0.0, 0.0);
		for (size_t t = 0; t < n; t++) {  // For each input element
			double angle = 2 * M_PI * t * k / n;
			sum += input[t] * exp(complex<double>(0, -angle));
		}
		output.push_back(sum);
	}
	return output;
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
}

int main(){
	for (int exp = 2; exp <= 9; exp++){
		u64 n = 2 << exp; // array size

		// Create random complex vector
		complex<double> *vec = new (std::align_val_t(32)) complex<double>[n];
		for (size_t i = 0; i < n; i++){
			vec[i] = complex<double>(1, 2);
			//cout << "vec["<< i << "] = " << vec[i] << endl;
		}
		/*
		for (size_t i = 0; i < n; i++){
			double angle = 2 * M_PI * i / n;
			complex<double> twiddle(1.0, 1.0);
			twiddle = twiddle * std::exp(complex<double>(0, -angle));
			cout << "twiddle["<< i << "] = " << twiddle << endl;
		}
		*/	
		// Calculate reference transform
		vector<complex<double> > ref = naiveDft(vec, n);
		/*
		for (size_t i = 0; i < n; i++){
			cout << "ref["<< i << "] = " << ref.at(i) << endl;
		}
		*/
		
		
		// calculate fft transform
		cf64* wave = new cf64[n];
		cf64* wave_tmp = new cf64[n];
		
		for(u64 i = 0; i < n; ++i){
			wave[i].re = vec[i].real();
			wave[i].im = vec[i].imag();
		}
		
		cf64* roots = new cf64[n/2];
		init_roots(roots,n);

		/*
		for (size_t i = 0; i < n; i++){
			cout << "vec["<< i << "] = " << re[i] << "," << im[i] << endl;
		}
		for (size_t i = 0; i < n; i++){
			cout << "twiddle["<< i << "] = " << cosines[i] << "," << sines[i] << endl;
		}
		*/
		fft_stockham(wave, wave_tmp, roots, n);
		/*
		for (size_t i = 0; i < n; i++){
			cout << "vec["<< i << "] = " << re_tmp[i] << "," << im_tmp[i] << endl;
		}
		*/

		if (exp % 2 != 0){

			for(u64 i = 0; i < n; ++i){
				
				vec[i] = complex<double>(wave[i].re, wave[i].im);
			}
		} else {
			for(u64 i = 0; i < n; ++i){
				
				vec[i] = complex<double>(wave_tmp[i].re, wave_tmp[i].im);
			}
		}

		// Calculate root-mean-square error
		double max_err = 0;
		for (size_t i = 0; i < n; i++){
			double tmp = std::norm(vec[i] - ref.at(i));
			if (tmp > max_err){
				max_err = tmp;
			}
		}
		cout << "N = " << n << " - max absolute error: " << max_err << endl;
		delete[] wave;
		delete[] wave_tmp;
		delete[] roots;
		delete[] vec;
	}
}

