#!/bin/sh

${main=./main} \
-AdaptSteps 20 \
-CFL 0.5 \
-Ctol 1 \
-lambda 1e7 \
-levelMax 7 \
-levelStart 4 \
-maxPoissonRestarts 0 \
-nu 0.0001 \
-poissonTol 1e-3 \
-poissonTolRel 1e-2 \
-Rtol 2 \
-tdump 0.1 \
-tend 8.0 \
-shapes '
   scale=0.125 orientation=0 omega=-0.3 xcenter=0.3 ycenter=0.5 sdf=box.raw
   scale=0.125 orientation=0 omega=0.3 xcenter=0.6 ycenter=0.5 sdf=box.raw
'
