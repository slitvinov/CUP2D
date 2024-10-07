d=/scratch/lisergey/CUP2D
ssh glados '
d='$d'
rm -rf ${d?not set} &&
   git clone git@github.com:slitvinov/CUP2D /scratch/lisergey/CUP2D &&
   cd "$d" &&
   git checkout '${1-HEAD}' &&
   module load mpi &&
   make -j "NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++" "CXXFLAGS = -O3" &&
   mpiexec -n 2 sh run.sh &&
   # ls vel.*.xdmf2 | xargs -n 1 -P `nproc --all` ./post.py
'
rsync -avz "glados:$d"/vel* .
