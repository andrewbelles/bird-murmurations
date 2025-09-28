#!/usr/bin/env make -f 
# 
# Makefile  Andrew Belles  Sept 28th 2025 
# 
# Root Makefile for project source code 
#
#

.PHONY: all clean 

all: 
	make -C simulations 

clean: 
	rm -f *~ 
	make -C simulations clean
