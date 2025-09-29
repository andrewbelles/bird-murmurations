#!/bin/bash 
#
# pipeline.sh  Andrew Belles Sept 26th, 2025 
#
# Full pipeline from cpp simulation to data analysis
# and simulation 
#

PYTHON=$(command -v python3 || command -v python)

./boids inputs.txt data/file.csv
$PYTHON analysis.py --file data/file.csv
$PYTHON visualize.py --file data/file.csv