#!/bin/sh

python3 gen_table.py &&
gcc-15 -O2 -o main main.c -fopenmp -lm &&
${main=./main} \
-AdaptSteps 2 \
-CFL 0.8 \
-Rtol 0.1 \
-levelMax 5 \
-levelStart 2 \
-nu 1e-4 \
-sdump 0 \
-tdump 0.4 \
-tend 1.2 \
&&
python3 make_amr_panels.py
