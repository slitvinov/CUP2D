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
make
```

```
-Wno-format-truncation\
-Wno-unused-result\
-Wno-cast-function-type\
-Wno-sign-compare\
```

debug:

```
main="xterm -e gdb -ex run -args ./main" sh run.sh
```

FAS RC:
```
module load gcc openmpi cuda
make
salloc -N 1 -n 2 -c 4 -p seas_gpu --gpus 1 --mem 1Gb
...
OMP_NUM_THREADS=4 main='srun --mpi=pmix ./main' sh -x run.sh
```

For hal/glados
```
module load mpi/openmpi-x86_64
git clean -fdxq && make 'CXXFLAGS = -coverage -Og -g3' NVCC='/usr/local/cuda-12.5/bin/nvcc -ccbin=mpic++' 'LDFLAGS = -Xcompiler -coverage' 'OPENMPFLAGS = '&& OMP_NUM_THREADS=2 mpiexec -n 2 sh run.sh && python3 tool/stat.py *.xdmf2 | tee ref.out && python -m gcovr --html-details cover.html
```

Paraview
```
for i in vel.*.xdmf2; do j=${i%.xdmf2}.png; if test ! -f $j; then echo $i $j; fi; done | xargs -r -P `nproc` -n 2 sh -xc 'pvbatch tool/view.py "$@"' sh
```
