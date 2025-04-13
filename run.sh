#!/bin/sh

${main=./main} \
-AdaptSteps 20 \
-CFL 0.5 \
-Ctol 1 \
-lambda 1e7 \
-levelMax 6 \
-levelStart 3 \
-maxPoissonRestarts 0 \
-nu 0.0001 \
-poissonTol 1e-3 \
-poissonTolRel 1e-2 \
-Rtol 2 \
-tdump 1 \
-tend 128 \
-shapes '
   scale=0.125 orientation=0 omega=-0.3 xcenter=0.375 ycenter=0.5 sdf=box.raw
   scale=0.125 orientation=0 omega=0.3 xcenter=0.525 ycenter=0.5 sdf=box.raw
'

# -shapes '
#   scale=0.75 orientation=0 omega=-0.05 xcenter=0.35 ycenter=0.55 sdf=sdf.raw
#   scale=0.75 orientation=0 omega=-0.05 xcenter=0.65 ycenter=0.44 sdf=sdf.raw
# '
