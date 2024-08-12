ssh glados '
cd /scratch/lisergey/CUP2D &&
   git clean -fdxq &&
   git pull &&
   module load mpi &&
   make "NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++" -j
   mpiexec -n 2 run.sh
'
scp -avz glados@/scratch/lisergey/CUP2D/vort* .
