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
		
		for(u64 i = 0; i < n; i += vector_step/2){
			// Skip if we're in the second half of a group
			// (corresponds to j >= halfp in original)
			u64 j_in_group = i % p;
			if(j_in_group >= halfp) {
				i += halfp;
				if(i >= n) break;
			}
			
			svbool_t buffer_pred = svwhilelt_b64(2*i, 2*n);
			
			// Create indices at complex level: i, i, i+1, i+1, i+2, i+2, ...
			svuint64_t indices = svindex_u64(2*i, 1);
			indices = svlsr_x(buffer_pred, indices, 1);
			
			// Keep only indices where (index % p) < halfp
			// (first half of each group)
			svuint64_t mod_p = svand_z(buffer_pred, indices, p-1);
			svbool_t valid_pred = svcmplt(buffer_pred, mod_p, halfp);
			
			// Calculate group and j for each element
			// group = index / p, j = index % p
			svuint64_t group_idx = svlsr_x(valid_pred, indices, logp);
			svuint64_t j = mod_p;
			
			// Calculate input indices: base_in_0 = group * halfp + j
			svuint64_t base_in_0 = svmla_x(valid_pred, j, group_idx, halfp);
			svuint64_t base_in_1 = svadd_x(valid_pred, base_in_0, n/2);
			
			// Calculate output indices: base_out = group * p + j
			svuint64_t base_out_even = svmla_x(valid_pred, j, group_idx, p);
			svuint64_t base_out_odd = svadd_x(valid_pred, base_out_even, halfp);
			
			// Convert complex indices to double indices (multiply by 2)
			svuint64_t double_in_0 = svlsl_n_u64_x(valid_pred, base_in_0, 1);
			svuint64_t double_in_1 = svlsl_n_u64_x(valid_pred, base_in_1, 1);
			svuint64_t double_out_even = svlsl_n_u64_x(valid_pred, base_out_even, 1);
			svuint64_t double_out_odd = svlsl_n_u64_x(valid_pred, base_out_odd, 1);
			
			// Load twiddle factors
			// root_indices pattern: j, j, j+1, j+1, ... scaled and interleaved
			svuint64_t root_indices = j;
			root_indices = svlsl_x(valid_pred, root_indices, logn-logp+1);  // *= step, *= 2
			root_indices = svorr_x(valid_pred, root_indices, sv_0101);  // interleave re/im
			svfloat64_t w = svld1_gather_index(valid_pred, roots_data, root_indices);
			
			// Load input values using gather (non-contiguous access)
			svfloat64_t val0 = svld1_gather_index(valid_pred, data_in, double_in_0);
			svfloat64_t val1 = svld1_gather_index(valid_pred, data_in, double_in_1);
			
			// Complex multiply: z = w * val1
			svfloat64_t z = svdup_f64(0);
			z = svcmla_x(valid_pred, z, w, val1, 0);
			z = svcmla_x(valid_pred, z, w, val1, 90);
			
			// Butterfly and scatter to output (non-contiguous access)
			svst1_scatter_index(valid_pred, data_out, double_out_even, svadd_x(valid_pred, val0, z));
			svst1_scatter_index(valid_pred, data_out, double_out_odd, svsub_x(valid_pred, val0, z));
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

