#!/bin/usr/env make -f  
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Makefile that handles compilation of all simulation source files 
# Contains calls for test as well 
#

.PHONY: all test clean 

all: 
	make -C network 
	make -C logger 

clean: 
	make -C network clean 
	make -C logger clean 

test: 
	make -C network test 
	make -C logger test  
