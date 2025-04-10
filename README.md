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
  void load(std::unordered_map<long long, TreeState> *tree,
            std::unordered_map<long long, Info *> *all, const Stencil &stencil,
            Info *info, bool applybc) {
    TreeState nei[3][3];
    Real *blocks[3][3][4];
    int xi, yi;
    int n = 1 << info->level;
    sfc_inverse(info->Z, info->level, &xi, &yi);
    bool xskin = xi == 0 || xi == n - 1;
    bool yskin = yi == 0 || yi == n - 1;
    int xskip = xi == 0 ? -1 : 1;
    int yskip = yi == 0 ? -1 : 1;
    for (int icode = 0; icode < 9; icode++) {
      int cx = icode % 3 - 1;
      int cy = icode / 3 - 1;
      if (cx == xskip && xskin)
        continue;
      if (cy == yskip && yskin)
        continue;
      if (cx == 1 && cy == 1)
        continue;
      TreeState state = nei[1 + cx][1 + cy] =
          (*tree)[sim.levels[info->level] + info->Znei[1 + cx][1 + cy]];
      switch (state) {
      case Active:
        break;
      case ParentIsActive:
        blocks[1 + cx][1 + cy][0] = (*all)[sim.levels[info->level - 1] +
                                           (info->Znei[1 + cx][1 + cy] >> 2)]
                                        ->block;
        break;
      case ChildrenAreActive:
        long long id = sim.levels[info->level + 1] + info->Znei[1 + cx][1 + cy];
        Info *nn = (*all)[id];
        blocks[1 + cx][1 + cy][0] =
            (*all)[sim.levels[info->level + 1] + nn->Znei[0][0]]->block;
        blocks[1 + cx][1 + cy][1] =
            (*all)[sim.levels[info->level + 1] + nn->Znei[0][1]]->block;
        blocks[1 + cx][1 + cy][2] =
            (*all)[sim.levels[info->level + 1] + nn->Znei[1][0]]->block;
        blocks[1 + cx][1 + cy][3] =
            (*all)[sim.levels[info->level + 1] + nn->Znei[1][1]]->block;
        break;
      }
    }
    load0(info->block, blocks, nei, all, stencil, info, applybc);
  }
```