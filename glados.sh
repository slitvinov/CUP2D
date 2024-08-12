ssh glados '
cd /scratch/lisergey/CUP2D &&
   git clean -fdxq &&
   git pull &&
   module load mpi &&
   make -j "NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++" "CXXFLAGS = -coverage -Og -g3" "LDFLAGS = -Xcompiler -coverage" "OPENMPFLAGS =" &&
   mpiexec -n 2 sh run.sh &&
   python -m gcovr --html-details cover.html &&
'
rsync -avz glados:/scratch/lisergey/CUP2D/vort* glados:/scratch/lisergey/CUP2D/cover* .
