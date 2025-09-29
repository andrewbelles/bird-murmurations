#!/bin/env/usr make -f 
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Compilation rules for all source files used for the communication 
# network that agents utilize in simulation
#

CC    = nvcc 
OBJS  = grid.o test_grid.o 
FLAGS = -O3 -std=c++17 -lineinfo -Xcompiler -fPIC -rdc=true --expt-extended-lambda 
VALGRIND = valgrind --leak-check=full --show-leak-kinds=all 

.PHONY: all clean test 

all: grid_test 

grid_test: $(OBJS)
	$(CC) $(FLAGS) $(OBJS) -o $@

grid.o: grid.cu grid.cuh 
	$(CC) $(FLAGS) -c $< -o $@

test_grid.o: test_grid.cpp grid.cuh 
	$(CC) $(FLAGS) -c $< -o $@

test: grid_test  
	./grid_test 

clean: 
	rm -f grid_test *.o 
