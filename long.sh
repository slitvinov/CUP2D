#!/bin/sh

${main=./main} \
-AdaptSteps 20 \
-CFL 0.5 \
-Ctol 1 \
-Rtol 2 \
-lambda 1e7 \
-levelMax 7 \
-levelStart 3 \
-maxPoissonRestarts 0 \
-nu 1e-4 \
-poissonTol 1e-3 \
-poissonTolRel 0 \
-tdump 0.5 \
-tend 100 \
-shapes '
   scale=0.06 orientation=0 omega=-1.12 xcenter=0.59 ycenter=0.53 sdf=box.raw
   scale=0.06 orientation=0 omega=1.54 xcenter=0.44 ycenter=0.68 sdf=box.raw
   scale=0.06 orientation=0 omega=-1.82 xcenter=0.44 ycenter=0.38 sdf=box.raw
   scale=0.06 orientation=0 omega=-1.92 xcenter=0.44 ycenter=0.53 sdf=box.raw
   scale=0.06 orientation=0 omega=1.98 xcenter=0.59 ycenter=0.38 sdf=box.raw
'
