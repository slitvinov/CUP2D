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
module load mpi
scl enable gcc-toolset-12 bash
PATH=$HOME/.local/bin:/usr/local/cuda-12.5/bin:$PATH
git clean -fdxq && make 'CXXFLAGS = -Og -g3' && mpirun -n 2 sh run.sh && python3 tool/stat.py *.xdmf2 | tee ref.out
```

or
```
PATH=$HOME/.local/bin:/usr/local/cuda-12.5/bin:$PATH && module load mpi && make
```

AddressSanitizer:
```
make 'NVCCFLAGS = -g -O0 -Xcompiler -fsanitize=address' \
     'CXXFLAGS = -O0 -g3 -fsanitize=address' \
     'LDFLAGS = -Xcompiler -fsanitize=address'
```
run with
```
ASAN_OPTIONS=protect_shadow_gap=0 sh run.sh
```

Paraview
```
for i in vel.*.xdmf2; do j=${i%.xdmf2}.png; if test ! -f $j; then echo $i $j; fi; done | xargs -r -P `nproc` -n 2 sh -xc 'pvbatch tool/view.py "$@"' sh
```
