hal/glados
```
module load mpi
make 'NVCC =/usr/local/cuda-12.5/bin/nvcc' -j
```

With code coverage
```
module load mpi
make 'NVCC =/usr/local/cuda-12.5/bin/nvcc' 'CXXFLAGS = -coverage -Og -g3' 'LDFLAGS = -Xcompiler -coverage' 'OPENMPFLAGS = ' -j
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
OMP_NUM_THREADS=4 main='srun ./main' sh -x run.sh
```

For hal/glados
```
scl enable gcc-toolset-12 bash
PATH=$HOME/.local/bin:/usr/local/cuda-12.5/bin:$PATH
git clean -fdxq && make 'CXXFLAGS = -Og -g3' && sh run.sh && python3 tool/stat.py *.xdmf2 | tee ref.out
```

or
```
PATH=$HOME/.local/bin:/usr/local/cuda-12.5/bin:$PATH && make
```

AddressSanitizer:
```
scl enable gcc-toolset-12 bash
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


```
struct ChildNeighborPattern {
  int cx, cy; // Direction of the neighbor
  int Bstep;  // Loop step: 1 (normal), 3 (diagonal), 4 (corner)
  int ys;     // Vertical stride step (usually 1 or 2)
  std::pair<int, int> child_offset[4]; // Relative (dx, dy) of children
  int count; // How many child_offset entries are valid
};
static constexpr ChildNeighborPattern childNeighborTable[] = {
    // cx, cy, Bstep, ys, children[], count
    {-1, -1, 3, 1, {{-1, -1}, {}, {}, {}}, 1},    // SW
    {0, -1, 1, 1, {{0, -1}, {1, -1}, {}, {}}, 2}, // S
    {1, -1, 3, 1, {{2, -1}, {}, {}, {}}, 1},      // SE
    {-1, 0, 1, 2, {{-1, 0}, {-1, 1}, {}, {}}, 2}, // W
    {1, 0, 1, 2, {{2, 0}, {2, 1}, {}, {}}, 2},    // E
    {-1, 1, 3, 1, {{-1, 2}, {}, {}, {}}, 1},      // NW
    {0, 1, 1, 1, {{0, 2}, {1, 2}, {}, {}}, 2},    // N
    {1, 1, 3, 1, {{2, 2}, {}, {}, {}}, 1},        // NE
};
const ChildNeighborPattern *get_child_pattern(int cx, int cy) {
  for (const auto &entry : childNeighborTable)
    if (entry.cx == cx && entry.cy == cy)
      return &entry;
  return NULL;
}
```