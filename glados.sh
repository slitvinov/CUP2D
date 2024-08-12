ssh glados 'cd /scratch/lisergey/CUP2D && git clean -fdxq && git pull && sh build.sh && sh run.sh'
scp -avz glados@/scratch/lisergey/CUP2D/vort* .
