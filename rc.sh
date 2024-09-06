d=/n/holyscratch01/koumoutsakos_lab/slitvinov/CUP2D
ssh rc '
. /etc/profile
d='$d' 
rm -rf "${d?not set}" &&
   git clone git@github.com:slitvinov/CUP2D "$d" &&
   cd "$d" &&
   git checkout '${1-HEAD}' &&
   module load gcc/12.2.0-fasrc01 openmpi cuda python &&
   # make -j "CXXFLAGS = -Wno-deprecated-declarations -Og -g3" -j &&
   make -j "CXXFLAGS = -coverage -Og -g3" "LDFLAGS = -Xcompiler -coverage" "OPENMPFLAGS = " &&
   OMP_NUM_THREADS=4 srun --mpi=pmix -p seas_gpu -c 1 -n 2 -N 1 --gpus 1 --mem 2Gb -t 30 sh -x run.sh &&
   # ls vort.*.xdmf2 | xargs -n 1 -P `nproc --all` ./post.py &&
   python -m gcovr --html-details cover.html
' &&
rsync -avz "rc:$d"/vort* "rc:$d"/cover* .
