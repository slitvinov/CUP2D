d=/n/netscratch/koumoutsakos_lab/Lab/slitvinov/remote.CUP2D
ssh rc '
. /etc/profile
d='$d' 
rm -rf "${d?not set}" &&
   git clone git@github.com:slitvinov/CUP2D "$d" &&
   cd "$d" &&
   git checkout '${1-HEAD}' &&
   module load gcc/13 openmpi cuda python &&
   # make -j "CXXFLAGS = -Wno-deprecated-declarations -Og -g3" -j &&
   make -j "CXXFLAGS = -coverage -Og -g3" "LDFLAGS = -Xcompiler -coverage" &&
   OMP_NUM_THREADS=4 srun --mpi=pmix -p gpu_test -c 1 -n 2 -N 1 -G 1 --mem 2Gb -t 30 sh -x run.sh &&
   source activate jepa
   ls vel.*.xdmf2 | xargs -n 1 -P `nproc --all` tool/post.py &&
   PYTHONNOUSERSITE= python -m gcovr --html-details cover.html
' &&
rsync -avz "rc:$d"/vel* "rc:$d"/cover* .
