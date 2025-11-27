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


// Private function prototypes
static vector<complex<double> > naiveDft(const complex<double> *input, size_t n);
void init_roots(f64* __restrict__ cosines, f64* __restrict__ sines, u64 n);
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
	assert((n & (n-1)) == 0 && n > 0);
	u64 logn = 63 - __builtin_clzll(n);
	u64 vector_step = svcntd();
	
	// Pointers for ping-pong buffering
	f64 *re_in = re, *im_in = im;
	f64 *re_out = re_tmp, *im_out = im_tmp;
	
	for(u64 logp = 1; logp <= logn; ++logp){
		u64 p = 1 << logp;
		u64 halfp = p >> 1;
		
		for(u64 i = 0; i < n; i += vector_step){
			// Skip if we're in the "odd half" of any group
			// This corresponds to j >= halfp in the original
			u64 j_in_group = i % p;
			if(j_in_group >= halfp) {
				i += halfp;
				if(i >= n) break;
			}
			
			svbool_t buffer_pred = svwhilelt_b64(i, n);
			svuint64_t indices = svindex_u64(i, 1);
			
			// Keep only indices where (index % p) < halfp
			// i.e., we're in the first half of each group
			svuint64_t mod_p = svand_z(buffer_pred, indices, p-1);
			svbool_t valid_pred = svcmplt(buffer_pred, mod_p, halfp);
			
			// Calculate which group each element belongs to
			svuint64_t group_idx = svlsr_n_u64_z(valid_pred, indices, logp);
			
			// j within the group (for twiddle factors)
			svuint64_t j = svand_x(valid_pred, indices, p-1);
			
			// Calculate input indices for Stockham pattern
			// base_in_0 = group * halfp + j
			// base_in_1 = base_in_0 + n/2
			svuint64_t base_in_0 = svmla_x(valid_pred, j, group_idx, halfp);
			svuint64_t base_in_1 = svadd_x(valid_pred, base_in_0, n/2);
			
			// Calculate output indices
			// base_out_even = group * p + j
			// base_out_odd = base_out_even + halfp
			svuint64_t base_out_even = svmla_x(valid_pred, j, group_idx, p);
			svuint64_t base_out_odd = svadd_x(valid_pred, base_out_even, halfp);
			
			// Load twiddle factors
			svuint64_t root_indices = svlsl_n_u64_m(valid_pred, j, logn-logp);
			svfloat64_t wre = svld1_gather_index(valid_pred, cosines, root_indices);
			svfloat64_t wim = svld1_gather_index(valid_pred, sines, root_indices);
			
			// Load input values using gather
			svfloat64_t re0 = svld1_gather_index(valid_pred, re_in, base_in_0);
			svfloat64_t im0 = svld1_gather_index(valid_pred, im_in, base_in_0);
			svfloat64_t re1 = svld1_gather_index(valid_pred, re_in, base_in_1);
			svfloat64_t im1 = svld1_gather_index(valid_pred, im_in, base_in_1);
			
			// Complex multiply with twiddle: z = w * val1
			svfloat64_t zre = svmul_x(valid_pred, wre, re1);
			zre = svmls_x(valid_pred, zre, wim, im1);
			svfloat64_t zim = svmul_x(valid_pred, wre, im1);
			zim = svmla_x(valid_pred, zim, wim, re1);
			
			// Butterfly and scatter to output
			svfloat64_t out_even_re = svadd_x(valid_pred, re0, zre);
			svfloat64_t out_even_im = svadd_x(valid_pred, im0, zim);
			svfloat64_t out_odd_re = svsub_x(valid_pred, re0, zre);
			svfloat64_t out_odd_im = svsub_x(valid_pred, im0, zim);
			
			svst1_scatter_index(valid_pred, re_out, base_out_even, out_even_re);
			svst1_scatter_index(valid_pred, im_out, base_out_even, out_even_im);
			svst1_scatter_index(valid_pred, re_out, base_out_odd, out_odd_re);
			svst1_scatter_index(valid_pred, im_out, base_out_odd, out_odd_im);
		}
		
		// Swap buffers for next iteration
		std::swap(re_in, re_out);
		std::swap(im_in, im_out);
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
		f64* re = new f64[n];
		f64* im = new f64[n];
		f64* re_tmp = new f64[n];
		f64* im_tmp = new f64[n];
		
		for(u64 i = 0; i < n; ++i){
			re[i] = vec[i].real();
			im[i] = vec[i].imag();
		}
		
		f64* cosines = new f64[n/2];
		f64* sines = new f64[n/2];
		init_roots(cosines, sines, n);

		/*
		for (size_t i = 0; i < n; i++){
			cout << "vec["<< i << "] = " << re[i] << "," << im[i] << endl;
		}
		for (size_t i = 0; i < n; i++){
			cout << "twiddle["<< i << "] = " << cosines[i] << "," << sines[i] << endl;
		}
		*/
		fft_stockham(re, im, re_tmp, im_tmp, cosines, sines, n);
		/*
		for (size_t i = 0; i < n; i++){
			cout << "vec["<< i << "] = " << re_tmp[i] << "," << im_tmp[i] << endl;
		}
		*/

		if (exp % 2 != 0){

			for(u64 i = 0; i < n; ++i){
				
				vec[i] = complex<double>(re[i], im[i]);
			}
		} else {
			for(u64 i = 0; i < n; ++i){
				
				vec[i] = complex<double>(re_tmp[i], im_tmp[i]);
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
		delete[] re;
		delete[] im;
		delete[] re_tmp;
		delete[] im_tmp;
		delete[] cosines;
		delete[] sines;
		delete[] vec;
	}
}

