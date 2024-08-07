.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
.SUFFIXES: .o

bs = 8
MPIC++ = mpic++
LIBS = -lgsl -lgslcblas -lhdf5
GSL_CFLAGS != pkg-config --cflags gsl
GSL_LDFLAGS != pkg-config --libs gsl
CXXFLAGS = -g -DNDEBUG -O3 -D_DOUBLE_PRECISION_ -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32 -I. -DDIMENSION=2 -fopenmp -DGPU_POISSON
S = \
source/ArgumentParser.cpp \
source/Obstacles/CarlingFish.cpp \
source/Obstacles/CStartFish.cpp \
source/Obstacles/CylinderNozzle.cpp \
source/Obstacles/ExperimentFish.cpp \
source/Obstacles/Fish.cpp \
source/Obstacles/FishData.cpp \
source/Obstacles/Naca.cpp \
source/Obstacles/NeuroKinematicFish.cpp \
source/Obstacles/ShapeLibrary.cpp \
source/Obstacles/ShapesSimple.cpp \
source/Obstacles/SmartCylinder.cpp \
source/Obstacles/SmartNaca.cpp \
source/Obstacles/StefanFish.cpp \
source/Obstacles/Teardrop.cpp \
source/Obstacles/Waterturbine.cpp \
source/Obstacles/Windmill.cpp \
source/Obstacles/ZebraFish.cpp \
source/Operators/AdaptTheMesh.cpp \
source/Operators/advDiff.cpp \
source/Operators/advDiffSGS.cpp \
source/Operators/ComputeForces.cpp \
source/Operators/Forcing.cpp \
source/Operators/Helpers.cpp \
source/Operators/PressureSingle.cpp \
source/Operators/PutObjectsOnGrid.cpp \
source/Poisson/AMRSolver.cpp \
source/Poisson/Base.cpp \
source/Shape.cpp \
source/Simulation.cpp \
source/SimulationData.cpp \
source/Utils/BufferedLogger.cpp \
source/Poisson/ExpAMRSolver.cpp \
source/Poisson/LocalSpMatDnVec.cpp \

C = \
source/Poisson/BiCGSTAB.cu \

OBJECTS = $(S:.cpp=.o) $(C:.cu=.o)
NVCC = nvcc
NVCCFLAGS = -arch=native -O3 -Xcompiler '$(CXXFLAGS)'
LIBS = -lcublas -lcusparse
all: simulation
simulation: source/main.o $(OBJECTS)
	$(LINK) -arch=native  main.o $(OBJECTS) $(LIBS) -o $@
.cpp.o:
	$(MPICXX) -o $@ -c $< $(CXXFLAGS) $(GSL_CFLAGS)
.cu.o:
	$(NVCCFLAGS) -o $@ -c $< $(NVCCFLAGS)
clean:
	rm -f simulation $(OBJECTS)
	rm -f *.o
