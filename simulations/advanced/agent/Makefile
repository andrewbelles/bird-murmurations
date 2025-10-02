#!/bin/env/usr make -f 
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Compilation rules for all source files used for instructions for boids agents  
# 
#

CC    = nvcc 
OBJS  = agent.o environment.o 
FLAGS = -O3 -std=c++17 
CUDAFLAGS = -lineinfo -Xcompiler -fPIC -rdc=true --expt-extended-lambda -Xcompiler -pthread 
VALGRIND = valgrind --leak-check=full --show-leak-kinds=all 
INCLUDES = -I../network/ -I../logger

.PHONY: all clean test 

all: $(OBJS) 

agent.o: agent.cu agent.cuh 
	$(CC) $(FLAGS) $(CUDAFLAGS) -c $< -o $@

environment.o: environment.cu environment.cuh 
	$(CC) $(FLAGS) $(CUDAFLAGS) -c $< -o $@

clean: 
	rm -f network_test *.o *.a  
