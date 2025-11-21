CXX = g++
CXXFLAGS = -static -Wall -std=c++20 -O2 -march=native \
            -Isrc -Isrc/utils -Isrc/algorithms/cooley-tuckey -Isrc/algorithms/stockham -Iconfig
VPATH = src src/algorithms/cooley-tuckey src/algorithms/stockham src/utils config

# Output directory
BUILD_DIR = build

# Executables to produce
TARGETS = \
	fft_cooley_bi_novector fft_cooley_bi_auto fft_cooley_bi_sve fft_cooley_bi_sve_3loop \
	fft_cooley_ci_novector fft_cooley_ci_auto fft_cooley_ci_sve fft_cooley_ci_sve_3loop \
	fft_stockham_bi_novector fft_stockham_bi_auto fft_stockham_bi_sve \
	fft_stockham_ci_novector fft_stockham_ci_auto fft_stockham_ci_sve \
	fft_stockham_bi_sve_2loop

# Prepend build/ to all target names
OUT_TARGETS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

# Default rule
all: $(OUT_TARGETS)

# ----------------------
# FFT build rules
# ----------------------

$(BUILD_DIR)/fft_cooley_bi_novector: fft_cooley_bi.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize $^ -o $@

$(BUILD_DIR)/fft_cooley_bi_auto: fft_cooley_bi.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DPRAGMA_VECTOR -fopt-info-vec $^ -o $@

$(BUILD_DIR)/fft_cooley_bi_sve: fft_cooley_bi_sve.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/fft_cooley_bi_sve_3loop: fft_cooley_bi_sve_3loop.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/fft_cooley_ci_novector: fft_cooley_ci.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize $^ -o $@

$(BUILD_DIR)/fft_cooley_ci_auto: fft_cooley_ci.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DPRAGMA_VECTOR -fopt-info-vec $^ -o $@

$(BUILD_DIR)/fft_cooley_ci_sve: fft_cooley_ci_sve.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/fft_cooley_ci_sve_3loop: fft_cooley_ci_sve_3loop.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/fft_stockham_bi_novector: fft_stockham_bi.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize $^ -o $@

$(BUILD_DIR)/fft_stockham_bi_auto: fft_stockham_bi.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DPRAGMA_VECTOR -fopt-info-vec $^ -o $@

$(BUILD_DIR)/fft_stockham_bi_sve: fft_stockham_bi_sve.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/fft_stockham_bi_sve_2loop: fft_stockham_bi_sve_2loop.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/fft_stockham_ci_novector: fft_stockham_ci.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize $^ -o $@

$(BUILD_DIR)/fft_stockham_ci_auto: fft_stockham_ci.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DPRAGMA_VECTOR -fopt-info-vec $^ -o $@

$(BUILD_DIR)/fft_stockham_ci_sve: fft_stockham_ci_sve.cpp timer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# ----------------------
# Utility rules
# ----------------------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
