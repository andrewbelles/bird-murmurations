#!/bin/bash 
# run_sim.sh  Andrew Belles  Oct 1st, 2025 
#
#
#
#
#

csv=logging.csv
cnt=256
time=1000

echo "[RUN] Running Simulation:"
echo -e "[RUN] ./simboids --env env.yaml --sim sim.yaml --logger $csv" 
echo -e "      --agents $cnt --bufr 32 --steps 1000 --noise 0.05 --loss 0.05"

# Run simulation 
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  ./simboids --env env.yaml --sim sim.yaml --logger "$csv" \
  --agents $cnt --bufr 32 --steps $time --noise 0.05 --loss 0.05

rc=$? 

if [[ "$rc" -ne 0 ]]; then 
  echo "[RUN] Exiting Render Early"
else 
  echo "[RUN] Running Analysis on Most Recent Simulation"
  ./summary.py --csv "$csv" 
  ls -1 *.png | head -n 1 | xargs -r xdg-open 
fi 
