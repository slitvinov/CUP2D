#!/bin/sh

${main=./main} \
-AdaptSteps 5 \
-CFL 0.7 \
-Ctol 0.05 \
-levelMax 7 \
-levelStart 2 \
-Rtol 0.5 \
-sdump 100 \
-tdump 0 \
-tend 0.01
