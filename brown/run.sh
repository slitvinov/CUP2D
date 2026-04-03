#!/bin/sh

gcc-15 -O2 -o main main.c -fopenmp -lm &&
${main=./main} \
-AdaptSteps 0 \
-CFL 0.8 \
-Ctol 0 \
-levelMax 3 \
-levelStart 3 \
-nu 1e-4 \
-Rtol 0 \
-sdump 0 \
-tdump 0.4 \
-tend 2.0
