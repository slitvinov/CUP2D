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
set -- vel.*.xyz.raw
for i; do ./mesh2png -i $i -o ${i/.xyz.raw/}.msh.png -s 1024 -w 5; done
for i; do ./mesh2iso -i $i -o ${i/.xyz.raw/}.iso.png -s 1024 -w 5; done
