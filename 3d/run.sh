#!/bin/sh
# Khokhlov Section 7.5: Shock-bubble interaction
# Domain 1 x 1/4 x 1/4, M=1.25 shock, bubble R=1/8
# Paper: 5 levels (l=5..9), finest cell 1/512
# Our mapping: levelStart=2 (l=5), levelMax=7 (max level 6 = l=9)

${main=./main} \
-AdaptSteps 5 \
-CFL 0.7 \
-Ctol 0.05 \
-levelMax 7 \
-levelStart 2 \
-Rtol 0.5 \
-sdump 5 \
-tdump 0 \
-tend 0.5
