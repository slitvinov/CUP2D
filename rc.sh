d=/n/holyscratch01/koumoutsakos_lab/slitvinov/CUP2D
ssh rc '
. /etc/profile
d='$d' 
rm -rf "${d?not set}" &&
   git clone git@github.com:slitvinov/CUP2D "$d" &&
   cd "$d" &&
   git checkout '${1-HEAD}' &&
   module load gcc openmpi cuda &&
   make -j "CXXFLAGS = -Wno-deprecated-declarations -Og -g3" -j &&
   OMP_NUM_THREADS=4 srun --mpi=pmix -p seas_gpu -c 1 -n 2 -N 1 --gpus 1 --mem 2Gb -t 15 sh -x run.sh
' &&
rsync -avz "rc:$d"/vort* .
