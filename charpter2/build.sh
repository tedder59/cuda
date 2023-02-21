#!/bin/bash
nvcc -ccbin g++ -m64 --std=c++11 --threads 0 -gencode arch=compute_86,code=compute_86 $@