#!/bin/bash

for cpp_file in *.cpp; do
    if [ -f "$cpp_file" ]; then
        exe_name="${cpp_file%.cpp}"
        output_file="result_${cpp_file%.cpp}.txt"
        
        if [[ "$cpp_file" == *sve* ]]; then
            g++ -march=native "$cpp_file" -o "$exe_name"
        else
            g++ "$cpp_file" -o "$exe_name"
        fi
        
        ./"$exe_name" > "$output_file"
    fi
done