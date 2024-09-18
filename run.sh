: ${main=./main}
$main \
-AdaptSteps 20 \
-bpdx 2 \
-bpdy 1 \
-CFL 0.5 \
-Ctol 1 \
-dt 0 \
-extent 4 \
-lambda 1e7 \
-levelMax 8 \
-levelStart 5 \
-maxPoissonIterations 1000 \
-maxPoissonRestarts 0 \
-nu 0.00004 \
-poissonTol 1e-3 \
-poissonTolRel 1e-2 \
-Rtol 2 \
-tdump 0.5 \
-tend 10.0 \
-shapes '
  angle=0 L=0.2 xpos=1.8 ypos=0.8
  angle=180 L=0.2 xpos=1.6 ypos=0.8
'
