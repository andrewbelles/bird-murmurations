#!/bin/usr/env make -f  
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Makefile that handles compilation of all simulation source files 
# Contains calls for test as well 
#

.PHONY: all test clean 

all: 
	make -C simple 
	make -C advanced   

clean: 
	make -C simple clean 
	make -C advanced clean 

test: 
	make -C simple test 
	make -C advanced test  
