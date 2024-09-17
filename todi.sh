d=/capstor/scratch/cscs/lisergey/CUP2D
ssh todi '
. /etc/profile
d='$d'
rm -rf "${d?not set}" &&
   git clone git@github.com:slitvinov/CUP2D "$d" &&
   cd "$d" &&
   git checkout '${1-HEAD}' &&
   module load cray &&
   module load nvidia &&
   make -j "CXXFLAGS = -Og -g3" "LDFLAGS = -Xcompiler -coverage" "OPENMPFLAGS = " &&
   OMP_NUM_THREADS=4 srun --mpi=pmix -p seas_gpu -c 1 -n 2 -N 1 --gpus 1 --mem 2Gb -t 30 sh -x run.sh
   # ls vort.*.xdmf2 | xargs -n 1 -P `nproc --all` ./post.py &&
   # python -m gcovr --html-details cover.html
' &&
rsync -avz "tod:$d"/vort* .
