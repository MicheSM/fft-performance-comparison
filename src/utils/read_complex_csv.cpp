#include "read_complex_csv.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

std::vector<std::complex<double>> read_complex_csv(const std::string& filename) {
    std::vector<std::complex<double>> result;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Error: could not open file " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double real, imag;
        char comma;

        if (ss >> real >> comma >> imag) {
            result.emplace_back(real, imag);
        }
    }

    return result;
}

int main() {
    std::vector<std::complex<double>> vec = read_complex_csv("data/inputs/complex_1024.txt");
    std::cout << "Read " << vec.size() << " complex numbers from the CSV file." << std::endl;
    std::cout << "First complex number: " << vec[0] << std::endl;
    return 0;
}