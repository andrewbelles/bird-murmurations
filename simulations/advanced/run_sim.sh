#!/bin/bash 
# run_sim.sh  Andrew Belles  Oct 1st, 2025 
#
# Bash script to run the simulation. I can't be fucked to 
# write this to take the args off command line so this is really 
# just because I am lazy 
#

./simboids --env env.yaml --sim sim.yaml --logger /dev/null \
	--agents 8 --bufr 32 --steps 3600000 --noise 0.05 --loss 0.05

./summary.py  

ls -1 *.png | head -n 1 | xargs -r xdg-open 
