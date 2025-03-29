d=/scratch/slitvinov/remote.CUP2D
ssh grace2 '
. /etc/profile
d='$d'
rm -rf "${d?not set}" &&
   git clone git@github.com:slitvinov/CUP2D "$d" &&
   cd "$d" &&
   git checkout '${1-HEAD}' &&
   MODULEPATH=/scratch/`whoami`/.grace/modulefiles:$MODULEPATH module load nvhpc/24.5 &&
   set -x &&
   make -j "CXXFLAGS = -O2 -g" "OPENMPFLAGS = " &&
   sh -x run.sh &&
   ls vel.*.xdmf2 | xargs -n 1 -P `nproc --all` ./post.py
' &&
rsync -avz "grace2:$d"/vel* .
