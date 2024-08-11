hal/glados
```
module load mpi
make 'NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++' -j
```

With code coverage
```
module load mpi
make 'NVCC =/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++' 'CXXFLAGS = -coverage -Og -g3' 'LDFLAGS = -Xcompiler -coverage' 'OPENMPFLAGS = ' -j
sh run.sh
python -m gcovr --html-details cover.html
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
make 'CXXFLAGS != pkg-config --cflags gsl hdf5-openmpi' 'LDFLAGS != pkg-config --libs gsl hdf5-openmpi' -j
```
