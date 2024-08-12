. /etc/profile
module load mpi
make 'NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++' -j

