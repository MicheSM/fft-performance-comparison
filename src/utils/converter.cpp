#include <stdexcept>
#include "converter.h"

// ========================================================
// complex to block-interleaved
// ========================================================
void complex_to_bi(
    const std::vector<std::complex<double>>& input,
    std::vector<double>& out_real,
    std::vector<double>& out_imag
) {
    size_t n = input.size();
    out_real.resize(n);
    out_imag.resize(n);

    for (size_t i = 0; i < n; i++) {
        out_real[i] = input[i].real();
        out_imag[i] = input[i].imag();
    }
}

// ========================================================
// complex to complex-interleaved (CI)
// ========================================================
std::vector<cf64> complex_to_ci(
    const std::vector<std::complex<double>>& input
) {
    size_t n = input.size();
    std::vector<cf64> out(n);

    for (size_t i = 0; i < n; i++) {
        out[i].re = input[i].real();
        out[i].im = input[i].imag();
    }

    return out;
}

// ========================================================
// block-interleaved to complex
// ========================================================
std::vector<std::complex<double>> bi_to_complex(
    const std::vector<double>& real,
    const std::vector<double>& imag
) {
    if (real.size() != imag.size()) {
        throw std::runtime_error("bi_to_complex: real/imag size mismatch");
    }

    size_t n = real.size();
    std::vector<std::complex<double>> out(n);

    for (size_t i = 0; i < n; i++) {
        out[i] = std::complex<double>(real[i], imag[i]);
    }

    return out;
}

// ========================================================
// complex-interleaved to complex
// ========================================================
std::vector<std::complex<double>> ci_to_complex(
    const std::vector<cf64>& input
) {
    size_t n = input.size();
    std::vector<std::complex<double>> out(n);

    for (size_t i = 0; i < n; i++) {
        out[i] = std::complex<double>(input[i].re, input[i].im);
    }

    return out;
}

// ========================================================
// block-interleaved to complex-interleaved
// ========================================================
std::vector<cf64> bi_to_ci(
    const std::vector<double>& real,
    const std::vector<double>& imag
) {
    if (real.size() != imag.size()) {
        throw std::runtime_error("bi_to_ci: real/imag size mismatch");
    }

    size_t n = real.size();
    std::vector<cf64> out(n);

    for (size_t i = 0; i < n; i++) {
        out[i].re = real[i];
        out[i].im = imag[i];
    }

    return out;
}

// ========================================================
// complex-interleaved to block-interleaved
// ========================================================
void ci_to_bi(
    const std::vector<cf64>& input,
    std::vector<double>& out_real,
    std::vector<double>& out_imag
) {
    size_t n = input.size();
    out_real.resize(n);
    out_imag.resize(n);

    for (size_t i = 0; i < n; i++) {
        out_real[i] = input[i].re;
        out_imag[i] = input[i].im;
    }
}
