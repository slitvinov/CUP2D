.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp .cu .o

NVCC = nvcc
MPICXX = mpic++
LIBS = -lgsl -lgslcblas -lhdf5

GSL_CFLAGS != pkg-config --cflags gsl
GSL_LDFLAGS != pkg-config --libs gsl

HDF_CFLAGS != pkg-config --cflags hdf5-openmpi
HDF_LDFLAGS != pkg-config --libs hdf5-openmpi

FLAGS = \
-D_BS_=8 \
-DCUBISM_ALIGNMENT=32 \
-D_DOUBLE_PRECISION_ \
-DGPU_POISSON -std=c++17 \
-DNDEBUG \
-I. -DDIMENSION=2 \
-O3 \

O = \
source/ArgumentParser.o \
source/main.o \
source/Obstacles/CarlingFish.o \
source/Obstacles/CStartFish.o \
source/Obstacles/CylinderNozzle.o \
source/Obstacles/ExperimentFish.o \
source/Obstacles/FishData.o \
source/Obstacles/Fish.o \
source/Obstacles/Naca.o \
source/Obstacles/NeuroKinematicFish.o \
source/Obstacles/ShapeLibrary.o \
source/Obstacles/ShapesSimple.o \
source/Obstacles/SmartCylinder.o \
source/Obstacles/SmartNaca.o \
source/Obstacles/StefanFish.o \
source/Obstacles/Teardrop.o \
source/Obstacles/Waterturbine.o \
source/Obstacles/Windmill.o \
source/Obstacles/ZebraFish.o \
source/Operators/AdaptTheMesh.o \
source/Operators/advDiff.o \
source/Operators/advDiffSGS.o \
source/Operators/ComputeForces.o \
source/Operators/Forcing.o \
source/Operators/Helpers.o \
source/Operators/PressureSingle.o \
source/Operators/PutObjectsOnGrid.o \
source/Poisson/AMRSolver.o \
source/Poisson/Base.o \
source/Poisson/BiCGSTAB.o \
source/Poisson/ExpAMRSolver.o \
source/Poisson/LocalSpMatDnVec.o \
source/Shape.o \
source/SimulationData.o \
source/Simulation.o \
source/Utils/BufferedLogger.o \

NVCCFLAGS =
main: $O
	$(MPICXX) -o main $O $(GSL_LDFLAGS) $(HDF_LDFLAGS) $(LDFLAGS) -fopenmp -lcublas -lcusparse
.cpp.o:
	$(MPICXX) -o $@ -c $< $(FLAGS) $(CXXFLAGS) $(GSL_CFLAGS) $(HDF_CFLAGS)
.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS) -arch=native -Xcompiler '$(FLAGS)'
clean:
	rm -f main $O
