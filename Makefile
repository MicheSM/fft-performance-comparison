CXX = g++
CXXFLAGS = -static -Wall -std=c++20 -O2 -mcpu=native
VPATH = src src/algorithms src/utils

all: fft_cooley_bi_novector   fft_cooley_bi_auto   fft_cooley_bi_sve   fft_cooley_bi_sve_3loop fft_cooley_ci_novector   fft_cooley_ci_auto   fft_cooley_ci_sve   fft_cooley_ci_sve_3loop fft_stockham_bi_novector fft_stockham_bi_auto fft_stockham_bi_sve fft_stockham_ci_novector fft_stockham_ci_auto fft_stockham_ci_sve

fft_cooley_bi_novector: fft_cooley_bi.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize fft_cooley_bi.cpp timer.cpp -o fft_cooley_bi_novector

fft_cooley_bi_auto: fft_cooley_bi.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) -DPRAGMA_VECTOR -fopt-info-vec fft_cooley_bi.cpp timer.cpp -o fft_cooley_bi_auto 

fft_cooley_bi_sve: fft_cooley_bi_sve.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) fft_cooley_bi_sve.cpp timer.cpp -o fft_cooley_bi_sve 

fft_cooley_bi_sve_3loop: fft_cooley_bi_sve_3loop.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) fft_cooley_bi_sve_3loop.cpp timer.cpp -o fft_cooley_bi_sve_3loop

fft_cooley_ci_novector: fft_cooley_ci.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize fft_cooley_ci.cpp timer.cpp -o fft_cooley_ci_novector 

fft_cooley_ci_auto: fft_cooley_ci.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) -DPRAGMA_VECTOR -fopt-info-vec fft_cooley_ci.cpp timer.cpp -o fft_cooley_ci_auto 

fft_cooley_ci_sve: fft_cooley_ci_sve.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) fft_cooley_ci_sve.cpp timer.cpp -o fft_cooley_ci_sve 

fft_cooley_ci_sve_3loop: fft_cooley_ci_sve_3loop.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) fft_cooley_ci_sve_3loop.cpp timer.cpp -o fft_cooley_ci_sve_3loop

fft_stockham_bi_novector: fft_stockham_bi.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize fft_stockham_bi.cpp timer.cpp -o fft_stockham_bi_novector 

fft_stockham_bi_auto: fft_stockham_bi.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) -DPRAGMA_VECTOR -fopt-info-vec fft_stockham_bi.cpp timer.cpp -o fft_stockham_bi_auto

fft_stockham_bi_sve:fft_stockham_bi_sve.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) fft_stockham_bi_sve.cpp timer.cpp -o fft_stockham_bi_sve

fft_stockham_ci_novector: fft_stockham_ci.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize fft_stockham_ci.cpp timer.cpp -o fft_stockham_ci_novector 

fft_stockham_ci_auto: fft_stockham_ci.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) -DPRAGMA_VECTOR -fopt-info-vec fft_stockham_ci.cpp timer.cpp -o fft_stockham_ci_auto 

fft_stockham_ci_sve: fft_stockham_ci_sve.cpp timer.cpp timer.h config.h
	$(CXX) $(CXXFLAGS) fft_stockham_ci_sve.cpp timer.cpp -o fft_stockham_ci_sve

clean:
	rm -f fft_cooley_bi_novector   fft_cooley_bi_auto   fft_cooley_bi_sve   fft_cooley_bi_sve_3loop \
	      fft_cooley_ci_novector   fft_cooley_ci_auto   fft_cooley_ci_sve   fft_cooley_ci_sve_3loop \
          fft_stockham_bi_novector fft_stockham_bi_auto fft_stockham_bi_sve \
          fft_stockham_ci_novector fft_stockham_ci_auto fft_stockham_ci_sve
