${main=./main} \
-AdaptSteps 20 \
-CFL 0.5 \
-Ctol 1 \
-lambda 1e7 \
-levelMax 8 \
-levelStart 5 \
-maxPoissonRestarts 0 \
-nu 0.00004 \
-poissonTol 1e-3 \
-poissonTolRel 1e-2 \
-Rtol 2 \
-tdump 0.5 \
-tend 10.0 \
-shapes '
  orientation=0 length=0.2 xcenter=0.3 ycenter=0.4
  orientation=90 length=0.2 xcenter=0.7 ycenter=0.6
'
