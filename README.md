hal/glados
```
module load mpi
make 'NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++' -j
```

With converage
```
module load mpi
make 'NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++' 'CXXFLAGS = -coverage -Og -g3' 'LDFLAGS = -Xcompiler -coverage' -j
sh run.sh
python -m gcovr --html-details main.html
```

grace
```
module purge
MODULEPATH=/scratch/`whoami`/.grace/modulefiles:$MODULEPATH module load nvhpc/24.5
module load mpi/openmpi-aarch64
make 'CXXFLAGS = -I/scratch/slitvinov/.grace/include' 'LDFLAGS = -L/scratch/slitvinov/.grace/lib -Xlinker -R/scratch/slitvinov/.grace/lib' -j
```

local
```
make "CXXFLAGS = `pkg-config --cflags hdf5-openmpi`"
```
