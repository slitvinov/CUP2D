#!/bin/sh

${main=./main} \
-AdaptSteps 5 \
-CFL 0.5 \
-Ctol 0.05 \
-levelMax 6 \
-levelStart 3 \
-Rtol 0.3 \
-tdump 0.005 \
-tend 0.04
