d=/n/netscratch/koumoutsakos_lab/Lab/slitvinov/remote.CUP2D
rsync -avz --exclude '.git' --exclude 'vel.*' --exclude '*.o' --exclude 'main' ./ "rc:$d/" &&
ssh rc '
. /etc/profile
d='$d'
cd "$d" &&
   module load gcc/13 openmpi cuda python &&
   make -j "CXXFLAGS = -coverage -Og -g3" "LDFLAGS = -Xcompiler -coverage" &&
   OMP_NUM_THREADS=4 srun --mpi=pmix -p gpu_test -c 1 -n 2 -N 1 -G 1 --mem 2Gb -t 30 sh -x run.sh
' &&
rsync -avz "rc:$d"/vel* "rc:$d"/cover* .
