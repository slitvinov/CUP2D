.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp .cu .o

MPICXX = mpic++
NVCC = nvcc -ccbin='$(MPICXX)'
LINK = $(NVCC)
LIBS = -lcublas -lcusparse -lhdf5 -lgsl -lgslcblas
OPENMPFLAGS = -fopenmp

FLAGS =\
-D_BS_=8\
-DCUBISM_ALIGNMENT=32\
-Wno-unknown-pragmas\
$(OPENMPFLAGS)\
-std=c++17\

O =\
cuda.o\
main.o\

NVCCFLAGS =
main: $O
	$(LINK) -o main $O $(LDFLAGS) -Xcompiler '$(OPENMPFLAGS)' $(LIBS)
.cpp.o:
	$(MPICXX) -c $< $(FLAGS) $(CXXFLAGS)
.cu.o:
	$(NVCC) -c $< $(NVCCFLAGS) -Xcompiler '$(FLAGS)'
clean:
	rm -f main $O
