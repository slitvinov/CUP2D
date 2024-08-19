d=/n/holyscratch01/koumoutsakos_lab/slitvinov/CUP2D
ssh rc '
. /etc/profile
module load gcc openmpi cuda &&
set -x
d='$d' 
rm -r "${d?not set}" &&
   git clone git@github.com:slitvinov/CUP2D "$d" &&
   cd "$d"
   module load gcc openmpi cuda python &&
   make -j "CXXFLAGS = -Wno-deprecated-declarations -coverage -Og -g3" "LDFLAGS = -Xcompiler -coverage" -j &&
   OMP_NUM_THREADS=4 srun --mpi=pmix -l -p seas_gpu -c 4 -n 2 -N 1 --gpus 1 --mem 1Gb sh -t 15 -x run.sh
   python -m gcovr --html-details cover.html
' &&
rsync -avz "rc:$d"/vort* "rc:$d"/cover* .
