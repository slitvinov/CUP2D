hal/glados
```
cd makefiles
module load mpi
make 'NVCC =/usr/local/cuda-11.8/bin/nvcc -I/usr/local/cuda-11.8/targets/x86_64-linux/include/' 'CXX = mpic++ -I/usr/local/cuda-11.8/targets/x86_64-linux/include -L/usr/local/cuda-12.5/targets/x86_64-linux/lib/' 'gpu = true' -k -j
```

grace
```
cd makefiles
module purge
MODULEPATH=/scratch/`whoami`/.grace/modulefiles:$MODULEPATH module load nvhpc/24.5
OMPI_CXX=g++ make  -j 'gpu = true' 'CXX = mpic++ -I/scratch/slitvinov/.grace/include -I/scratch/slitvinov/.grace/Linux_aarch64/24.5/cuda/12.4/targets/sbsa-linux/include' LINK=nvcc
```
