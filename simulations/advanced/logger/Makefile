#!/bin/env/usr make -f 
#
# Makefile  Andrew Belles  Sept 29th, 2025 
#
# Compilation rules for data logger which sends information about agents positions 
# and velocities to the CPU, then writes to csv 
#

CC    = nvcc 
OBJS  = logger.o test_logger.o 
FLAGS = -O3 -std=c++17 
CUDAFLAGS = -lineinfo -Xcompiler -fPIC -rdc=true --expt-extended-lambda -Xcompiler -pthread 
VALGRIND = valgrind --leak-check=full --show-leak-kinds=all 

.PHONY: all clean test 

all: logger_test $(OBJS) 

logger_test: $(OBJS)
	$(CC) $(FLAGS) $(CUDAFLAGS) $(OBJS) -o $@

logger.o: logger.cu logger.cuh 
	$(CC) $(FLAGS) $(CUDAFLAGS) -c $< -o $@

test_logger.o: test_logger.cu logger.cuh
	$(CC) $(FLAGS) $(CUDAFLAGS) -c $< -o $@

test: logger_test  
	./logger_test test.bin 

clean: 
	rm -f logger_test *.o *.a *.bin 
