${main=./main} \
-AdaptSteps 20 \
-CFL 0.5 \
-Ctol 1 \
-lambda 1e7 \
-levelMax 7 \
-levelStart 4 \
-maxPoissonRestarts 0 \
-nu 1e-4 \
-poissonTol 1e-3 \
-poissonTolRel 1e-2 \
-Rtol 0.1 \
-tdump 0.25 \
-tend 600 \
-shapes '
   scale=0.125 orientation=0 omega=-1.82 xcenter=0.44 ycenter=0.38 sdf=blob.raw
   scale=0.125 orientation=0 omega=-1.92 xcenter=0.44 ycenter=0.53 sdf=blob.raw
   scale=0.125 orientation=0 omega=1.54 xcenter=0.44 ycenter=0.68 sdf=blob.raw
   scale=0.125 orientation=0 omega=1.98 xcenter=0.59 ycenter=0.38 sdf=blob.raw
   scale=0.125 orientation=0 omega=-1.12 xcenter=0.59 ycenter=0.53 sdf=blob.raw
'
