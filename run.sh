#!/bin/sh

${main=./main} \
-AdaptSteps 20 \
-CFL 0.5 \
-Ctol 1 \
-levelMax 6 \
-levelStart 3 \
-maxPoissonRestarts 0 \
-nu 0.0001 \
-poissonTol 1e-3 \
-poissonTolRel 1e-2 \
-Rtol 2 \
-tdump 0.125 \
-tend 1.0
