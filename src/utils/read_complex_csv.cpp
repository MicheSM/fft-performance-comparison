#include <vector>
#include <complex>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<std::complex<double>> read_complex_csv(const std::string& filename, size_t length) {
    std::vector<std::complex<double>> result;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    size_t count = 0;
    
    while (std::getline(file, line) && count < length) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        double real, imag;
        char comma;
        
        if (iss >> real >> comma >> imag) {
            result.emplace_back(real, imag);
            count++;
        }
    }
    
    file.close();
    return result;
}