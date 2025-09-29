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
DATA := data

.PHONY: all clean run test remake

all: $(OBJS) | $(DATA)

boids: boids.o | $(DATA)
	$(CC) $(FLAGS) $^ -o $@ 

boids.o: boids.cpp | $(DATA)
	$(CC) $(FLAGS) -c $< -o $@ 

$(DATA):
	mkdir -p $(DATA)

test: boids | $(DATA)
	$(VALGRIND) ./boids inputs.txt $(DATA)/file.csv

run: boids 
	./pipeline.sh

clean: 
	rm -f boids *.o
	rm -rf $(DATA)

remake: clean all
	@echo "Project cleaned and rebuilt successfully."