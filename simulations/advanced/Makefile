#!/bin/usr/env make -f  
#
# Makefile  Andrew Belles  Sept 28th, 2025 
#
# Makefile that handles compilation of all simulation source files 
# Contains calls for test as well 
#

CUDA = nvcc 
FLAGS = -O3 -std=c++17 -lineinfo -rdc=true --expt-extended-lambda -lcudart  
INCLUDES = -I. -Inetwork -Ilogger -Iagent
OBJS = main.o simulation.o 

NETWORK_OBJS = network/mailbox.o network/grid.o network/communication.o  
LOGGER_OBJS  = logger/logger.o 
AGENT_OBJS   = agent/agent.o agent/environment.o  
VALGRIND = valgrind --leak-check=full --show-leak-kinds=all 
LIBS = -Lagent -lagent -Lnetwork -lnetwork -Llogger -llogger 

.PHONY: all test clean run $(NETWORK_OBJS) $(LOGGER_OBJS) $(AGENT_OBJS) 

all: simboids 

dlink.o: $(OBJS) $(AGENT_OBJS) $(NETWORK_OBJS) $(LOGGER_OBJS)
	$(CUDA) $(FLAGS) $(INCLUDES) -dlink $^ -o $@

simboids: $(OBJS) $(AGENT_OBJS) $(NETWORK_OBJS) $(LOGGER_OBJS) dlink.o 
	$(CUDA) $(FLAGS) $(INCLUDES) $^ -o $@ 

main.o: main.cu simulation.cuh agent/environment.cuh agent/agent.cuh 
	$(CUDA) $(FLAGS) $(INCLUDES) -c $< -o $@

simulation.o: simulation.cu simulation.cuh network/grid.cuh network/mailbox.cuh network/communication.cuh logger/logger.cuh agent/environment.cuh agent/agent.cuh  
	$(CUDA) $(FLAGS) $(INCLUDES) -c $< -o $@

$(NETWORK_OBJS):
	make -C network 

$(LOGGER_OBJS):
	make -C logger 

$(AGENT_OBJS):
	make -C agent 

clean: 
	rm -f $(OBJS) simboids *.o *.csv *.png  
	make -C network clean 
	make -C logger clean 
	make -C agent clean 

test: 
	make -C network test 
	make -C logger test  

run: 
	./simboids --env env.yaml --sim sim.yaml --logger logging.csv \
		--agents 1024 --bufr 32 --steps 10000 
