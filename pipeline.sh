#!/bin/bash 
#
# pipeline.sh  Andrew Belles Sept 26th, 2025 
#
# Full pipeline from cpp simulation to data analysis
# and simulation 
#

./compile.sh 

./boids inputs.txt data/file.csv  

python sim_analysis.py --file data/file.csv 

python visualize.py --file data/file.csv
