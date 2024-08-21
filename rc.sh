d=/n/holyscratch01/koumoutsakos_lab/slitvinov/CUP2D
ssh rc '
. /etc/profile
module load gcc openmpi cuda &&
set -x
d='$d' 
rm -rf "${d?not set}" &&
   git clone git@github.com:slitvinov/CUP2D "$d" &&
   cd "$d"
   module load gcc openmpi cuda python &&
   make -j "CXXFLAGS = -Wno-deprecated-declarations -Og -g3" -j &&
   OMP_NUM_THREADS=4 srun --mpi=pmix -l -p seas_gpu -c 1 -n 2 -N 1 --gpus 1 --mem 1Gb -t 15 sh -x run.sh
' &&
rsync -avz "rc:$d"/vort* .
