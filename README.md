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
make 'gpu = true' 'CXX = OMPI_CXX=g++ mpic++ -I/scratch/slitvinov/.grace/include -I/scratch/slitvinov/.grace/Linux_aarch64/24.5/cuda/12.4/targets/sbsa-linux/include' 'LINK = nvcc -ccbin=mpic++ -L/scratch/slitvinov/.grace/lib -Xlinker -R/scratch/slitvinov/.grace/lib/' NVCC='nvcc -ccbin=mpic++' -j
```
