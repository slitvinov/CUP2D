#!/bin/sh

gcc-15 -O2 -o main main.c -fopenmp -lm &&
    ./main -levelStart 3 -levelMax 3 -AdaptSteps 0 -Rtol 0 -Ctol 0 -nu 1e-4 -CFL 0.8 -tend 2.0 -tdump 0.4
    
