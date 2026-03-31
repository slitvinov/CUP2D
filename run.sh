#!/bin/sh

${main=./main} \
-AdaptSteps 5 \
-CFL 0.7 \
-Ctol 0.35 \
-levelMax 8 \
-levelStart 3 \
-Rtol 0.5 \
-sdump 100 \
-tdump 0 \
-tend 0.01
