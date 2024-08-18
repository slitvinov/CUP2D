: ${main=./main}
$main -bpdx 2 -bpdy 1 -levelMax 8 -levelStart 5 -Rtol 2 -Ctol 1 -extent 4 -CFL 0.5 -poissonTol 1e-3 -poissonTolRel 1e-2 -maxPoissonRestarts 0 -bAdaptChiGradient 0 -tdump 0.5 -nu 0.00004 -tend 10.0 -shapes '
       	       	 stefanfish L=0.2 T=1 xpos=1.8 ypos=0.8
                 stefanfish L=0.2 T=1 xpos=1.6 ypos=0.8 angle=180
'
