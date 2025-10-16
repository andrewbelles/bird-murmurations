#!/bin/env/usr make -f 
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Compilation rules for render backend allowing for real time visualization of 
# simulation 
#

CC    = nvcc 
OBJS  = render.o 
FLAGS = -O3 -std=c++17 -lineinfo -Xcompiler -fPIC -rdc=true --expt-extended-lambda 
VALGRIND = valgrind --leak-check=full --show-leak-kinds=all 
INCLUDES = -I. -I.. 

.PHONY: all clean 

all: $(OBJS)  

render.o: render.cu render.cuh  
	$(CC) $(FLAGS) $(INCLUDES) -c $< -o $@

clean: 
	rm -f $(OBJS)  
