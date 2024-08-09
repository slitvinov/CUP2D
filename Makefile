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
-DDIMENSION=2\
-D_DOUBLE_PRECISION_\
-DGPU_POISSON\
-DNDEBUG\
$(OPENMPFLAGS)\
-I.\
-O3\
-std=c++17\


O =\
AdaptTheMesh.o\
advDiff.o\
ArgumentParser.o\
Base.o\
BiCGSTAB.o\
BufferedLogger.o\
ComputeForces.o\
ExpAMRSolver.o\
Helpers.o\
LocalSpMatDnVec.o\
main.o\
PressureSingle.o\
PutObjectsOnGrid.o\
Shape.o\
SimulationData.o\

NVCCFLAGS =
main: $O
	$(LINK) -o main $O $(LDFLAGS) -Xcompiler $(OPENMPFLAGS) $(LIBS)
.cpp.o:
	$(MPICXX) -c $< $(FLAGS) $(CXXFLAGS)
.cu.o:
	$(NVCC) -c $< $(NVCCFLAGS) -Xcompiler '$(FLAGS)'
clean:
	rm -f main $O
