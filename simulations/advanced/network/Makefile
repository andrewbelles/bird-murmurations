#!/bin/env/usr make -f 
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Compilation rules for all source files used for the communication 
# network that agents utilize in simulation
#

CC    = nvcc 
OBJS  = grid.o mailbox.o communication.o test_network.o 
FLAGS = -O3 -std=c++17 -lineinfo -Xcompiler -fPIC -rdc=true --expt-extended-lambda 
VALGRIND = valgrind --leak-check=full --show-leak-kinds=all 

.PHONY: all clean test 

all: network_test $(OBJS)  

network_test: $(OBJS)
	$(CC) $(FLAGS) $(OBJS) -o $@

grid.o: grid.cu grid.cuh 
	$(CC) $(FLAGS) -c $< -o $@

mailbox.o: mailbox.cu mailbox.cuh 
	$(CC) $(FLAGS) -c $< -o $@

communication.o: communication.cu communication.cuh grid.cuh mailbox.cuh 
	$(CC) $(FLAGS) -c $< -o $@

test_network.o: test_network.cpp grid.cuh mailbox.cuh communication.cuh 
	$(CC) $(FLAGS) -c $< -o $@

test: network_test  
	./network_test 

clean: 
	rm -f network_test *.o *.a  
