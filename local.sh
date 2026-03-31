python3 gen_table.py &&
make clean &&
make -j CC=clang \
  "OPENMPFLAGS=-Xpreprocessor -fopenmp" \
  "CFLAGS=-O2 -g -I/opt/homebrew/opt/libomp/include" \
  "LDFLAGS=-L/opt/homebrew/opt/libomp/lib -lomp" &&
OMP_NUM_THREADS=4 sh run.sh
