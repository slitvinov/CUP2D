ssh glados '
rm -rf /scratch/lisergey/CUP2D &&
   git clone git@github.com:slitvinov/CUP2D /scratch/lisergey/CUP2D &&
   cd /scratch/lisergey/CUP2D
   module load mpi &&
   make "NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++" "CXXFLAGS = -coverage -Og -g3" "LDFLAGS = -Xcompiler -coverage" "OPENMPFLAGS = " -j
   mpiexec -n 2 sh ~/run.sh &&
   python -m gcovr --html-details cover.html
'
rsync -avz glados:/scratch/lisergey/CUP2D/vort* glados:/scratch/lisergey/CUP2D/cover* .
