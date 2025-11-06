#include <chrono>
#include <string>
#include <iostream>
#include "timer.h"
#include "config.h"

#ifdef ENABLE_TIMERS
timer::timer(){
    tlast = t0 = std::chrono::high_resolution_clock::now();
}

void timer::print_time(std::string description){
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t1_minus_t0 = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    auto t1_minus_tlast = std::chrono::duration_cast<std::chrono::microseconds>(t1-tlast).count();
#ifdef COMPACT_OUTPUT
    std::cout << t1_minus_tlast << " " << t1_minus_t0 << " ";
#else
    if(description != "") description += ": ";
    std::cout << description << t1_minus_tlast << " us since last split, " << t1_minus_t0 << " us since start\n";
#endif
    tlast = t1;
}

timer::~timer(){
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t1_minus_t0 = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
#ifdef COMPACT_OUTPUT
    std::cout << t1_minus_t0 << "\n";
#else
    std::cout << "finished " << t1_minus_t0 << " us since start\n";
#endif
}
#else
timer::timer(){}
void timer::print_time(std::string description){}
timer::~timer(){}
#endif
