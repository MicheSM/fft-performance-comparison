#include <vector>
#include <complex>

typedef double f64;

struct cf64 {
    f64 re;
    f64 im;
};


// complex to block interleaved
void complex_to_bi(
    const std::vector<std::complex<double>>& input,
    std::vector<double>& out_real,
    std::vector<double>& out_imag
);

// complex to complex interleaved
std::vector<cf64> complex_to_ci(
    const std::vector<std::complex<double>>& input
);

// block interleaved to complex
std::vector<std::complex<double>> bi_to_complex(
    const std::vector<double>& real,
    const std::vector<double>& imag
);

// complex interleaved to complex
std::vector<std::complex<double>> ci_to_complex(
    const std::vector<cf64>& input
);

// block interleaved to complex interleaved
std::vector<cf64> bi_to_ci(
    const std::vector<double>& real,
    const std::vector<double>& imag
);

// complex interleaved to block interleaved
void ci_to_bi(
    const std::vector<cf64>& input,
    std::vector<double>& out_real,
    std::vector<double>& out_imag
);