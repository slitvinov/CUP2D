#!/bin/sh

${main=./main} \
-AdaptSteps 5 \
-CFL 0.7 \
-Ctol 0.05 \
-levelMax 7 \
-levelStart 3 \
-Rtol 0.5 \
-tdump 1.25e-4 \
-tend 1e-3
