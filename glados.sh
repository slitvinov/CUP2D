ssh glados '
rm -rf /scratch/lisergey/CUP2D &&
   git clone git@github.com:slitvinov/CUP2D /scratch/lisergey/CUP2D &&
   cd /scratch/lisergey/CUP2D
   module load mpi &&
   make "NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++" "CXXFLAGS = -O3" "OPENMPFLAGS = " -j
   mpiexec -n 2 sh ~/run.sh
'
rsync -avz glados:/scratch/lisergey/CUP2D/vort* .
