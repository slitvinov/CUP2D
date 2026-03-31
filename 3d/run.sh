#!/bin/sh

${main=./main} \
-AdaptSteps 5 \
-CFL 0.5 \
-Ctol 0.05 \
-levelMax 5 \
-levelStart 2 \
-Rtol 0.5 \
-sdump 10 \
-tdump 0 \
-tend 0.003
