${main=./main} \
-AdaptSteps 20 \
-CFL 0.5 \
-Ctol 1 \
-lambda 1e7 \
-levelMax 8 \
-levelStart 6 \
-maxPoissonRestarts 0 \
-nu 0.00004 \
-poissonTol 1e-3 \
-poissonTolRel 1e-2 \
-Rtol 2 \
-tdump 0.5 \
-tend 10.0 \
-shapes '
  sdf=sdf.raw orientation=90 omega_fixed=-0.4 length=0.50 xcenter=0.4 ycenter=0.5
  sdf=sdf.raw orientation=90 omega_fixed=+0.4 length=0.50 xcenter=0.6 ycenter=0.5
'
