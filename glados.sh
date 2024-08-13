ssh glados '
rm -rf /scratch/lisergey/CUP2D &&
   git clone git@github.com:slitvinov/CUP2D /scratch/lisergey/CUP2D &&
   cd /scratch/lisergey/CUP2D
   module load mpi &&
   make -j "NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++" "CXXFLAGS = -O3" "OPENMPFLAGS =" &&
   mpiexec -n 2 sh ~/run.sh
   python -m gcovr --html-details cover.html
'
rsync -avz glados:/scratch/lisergey/CUP2D/vort* glados:/scratch/lisergey/CUP2D/cover* .
