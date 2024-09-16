: ${main=./main}
$main \
-bAdaptChiGradient 0 \
-bpdx 2 \
-bpdy 1 \
-CFL 0.5 \
-Ctol 1 \
-extent 4 \
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
       	       	 stefanfish L=0.2 T=1 xpos=1.8 ypos=0.8
                 stefanfish L=0.2 T=1 xpos=1.6 ypos=0.8 angle=180
'
