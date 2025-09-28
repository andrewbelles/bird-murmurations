#!/bin/env/usr make -f 
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Makefile handling compilation of threaded cpp 
# simulation for linear dynamics + simple communication overhead boids 
#

CC    = clang++ 
OBJS  = boids.o 
FLAGS = -Wall -Wextra -Werror -O3 -pedantic -std=c++20 -pthread -ggdb 
VALGRIND = valgrind --leak-check=full --show-leak-kinds=all 

.PHONY: all clean run 

all: $(OBJS)

boids: boids.o 
	$(CC) $(FLAGS) $^ -o $@ 

boids.o: boids.cpp 
	$(CC) $(FLAGS) -c $< -o $@ 

test: boids 
	$(VALGRIND) ./boids input.txt data/file.csv

run: boids 
	./pipeline.sh

clean: 
	rm -f boids *.o 
