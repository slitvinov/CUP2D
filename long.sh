#!/bin/sh

${main=./main} \
-AdaptSteps 20 \
-CFL 0.5 \
-Ctol 0.01 \
-lambda 1e7 \
-levelMax 9 \
-levelStart 4 \
-maxPoissonRestarts 0 \
-nu 1e-4 \
-poissonTol 1e-3 \
-poissonTolRel 0 \
-Rtol 0.1 \
-tdump 0.1 \
-tend 600 \
-shapes '
   scale=0.06 orientation=0 omega=-1.82 xcenter=0.44 ycenter=0.38 sdf=box.raw
   scale=0.06 orientation=0 omega=-1.92 xcenter=0.44 ycenter=0.53 sdf=box.raw
   scale=0.06 orientation=0 omega=1.54 xcenter=0.44 ycenter=0.68 sdf=box.raw
   scale=0.06 orientation=0 omega=1.98 xcenter=0.59 ycenter=0.38 sdf=box.raw
   scale=0.06 orientation=0 omega=-1.12 xcenter=0.59 ycenter=0.53 sdf=box.raw
'

