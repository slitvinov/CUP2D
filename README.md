hal/glados
```
module load mpi
make 'NVCC =/usr/local/cuda-11.8/bin/nvcc -I/usr/local/cuda-11.8/targets/x86_64-linux/include/' 'CXX = mpic++ -I/usr/local/cuda-11.8/targets/x86_64-linux/include -L/usr/local/cuda-12.5/targets/x86_64-linux/lib/' 'gpu = true' -k -j
```

grace
```
module purge
MODULEPATH=/scratch/`whoami`/.grace/modulefiles:$MODULEPATH module load nvhpc/24.5
module load mpi/openmpi-aarch64
make 'CXXFLAGS = -I/scratch/slitvinov/.grace/include' 'LDFLAGS = -L/scratch/slitvinov/.grace/lib -Xlinker -R/scratch/slitvinov/.grace/lib' -j -B
```

local
```
make "CXX = mpic++.openmpi `pkg-config --cflags hdf5-openmpi`"
```
