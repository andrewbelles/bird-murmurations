#!/bin/env/usr make -f 
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Compilation rules for all source files used for instructions for boids agents  
# 
#

CC    = nvcc 
OBJS  = agent.o environment.o # test_agent.o 
FLAGS = -O3 -std=c++17 
CUDAFLAGS = -lineinfo -Xcompiler -fPIC -rdc=true --expt-extended-lambda -Xcompiler -pthread 
VALGRIND = valgrind --leak-check=full --show-leak-kinds=all 
INCLUDES = -I../network/ -I../logger

.PHONY: all clean test 

all: agent_test 

agent.o: agent.cu agent.cuh 
	$(CC) $(FLAGS) $(CUDAFLAGS) -c $< -o $@

environment.o: environment.cu environment.cuh 
	$(CC) $(FLAGS) $(CUDAFLAGS) -c $< -o $@

test_agent.o: test_agent.cu agent.cuh environment.cuh
	$(CC) $(FLAGS) $(CUDAFLAGS) -c $< -o $@
	
agent_test: $(OBJS)
	$(CC) $(FLAGS) $(CUDAFLAGS) $(OBJS) -o $@

test: agent_test  
	./network_test 

clean: 
	rm -f network_test *.o 
