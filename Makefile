.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp .cu .o

MPICXX = mpicxx
NVCC = nvcc -ccbin='$(MPICXX)'
LINK = $(NVCC)
LIBS = -lcublas -lcusparse -lgsl -lgslcblas
OPENMPFLAGS = -fopenmp

FLAGS =\
-D_BS_=8\
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
main: kernel.h

