#include <chrono>
#include <string>

class timer{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t0;
    std::chrono::time_point<std::chrono::high_resolution_clock> tlast;
public:
    timer();
    void print_time(std::string description = "");
    ~timer();
};
