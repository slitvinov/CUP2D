#!/bin/sh

python3 gen_table.py &&
cc -O2 -fsanitize=address,undefined -fno-omit-frame-pointer -o main main.c -lm &&
cc -O2 -o mesh2png ../tools/mesh2png.c -lz &&
cc -O2 -o mesh2iso ../tools/mesh2iso.c -lz -lm &&
${main=./main} \
-AdaptSteps 0 \
-CFL 0.9 \
-Rtol 0.1 \
-levelMax 3 \
-levelStart 3 \
-nu 1e-4 \
-sdump 0 \
-tdump 0.4 \
-tend 1.2 \
&&
set -- *.xyz.raw
for i; do ./mesh2png -i "$i" -o "${i%.xyz.raw}.msh.png" -s 1024 -w 3; done
for i; do ./mesh2iso -g "$i" -s "${i%.xyz.raw}.vort.raw" -o "${i%.xyz.raw}.iso.png" -n 1024 -w 3 < levels.txt; done
