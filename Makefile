.POSIX:
.SUFFIXES:
.SUFFIXES: .c .cpp .cu .o

CXX = g++
CC = gcc
CXXFLAGS = -O2 -g
CFLAGS = -O2 -g
NVCC = nvcc
OPENMPFLAGS ?= -fopenmp

ifdef CPU
O = solver_cpu.o main.o
LINK = $(CXX)
LDFLAGS_EXTRA = $(OPENMPFLAGS) -lm
else
O = solver_gpu.o main.o
LINK = $(NVCC)
LDFLAGS_EXTRA = -Xcompiler '$(OPENMPFLAGS)' -lcublas -lcusparse
endif

main: $O
	$(LINK) -o main $O $(LDFLAGS) $(LDFLAGS_EXTRA)
.c.o:
	$(CC) -c $< $(OPENMPFLAGS) $(CFLAGS)
.cpp.o:
	$(CXX) -c $< $(OPENMPFLAGS) $(CXXFLAGS)
.cu.o:
	$(NVCC) -c $< $(NVCCFLAGS)
clean:
	rm -f main $O solver_cpu.o solver_gpu.o

main.o: utils.h solver.h
solver_cpu.o: solver.h
solver_gpu.o: solver.h
