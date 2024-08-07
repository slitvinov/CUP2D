bs = 8
CXX = mpic++
LINK = $(CXX)
LIBS = -lgsl -lgslcblas -lhdf5
CPPFLAGS = -g -DNDEBUG -O3 -D_DOUBLE_PRECISION_
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32
CPPFLAGS += -I. -DDIMENSION=2
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
OBJECTS += ExpAMRSolver.o BiCGSTAB.o LocalSpMatDnVec.o
CPPFLAGS += -fopenmp -DGPU_POISSON
NVCCFLAGS += -arch=native -O3 -Xcompiler '$(CPPFLAGS)' -DGPU_POISSON
LIBS += -lcublas -lcusparse
all: simulation
simulation: source/main.o $(OBJECTS)
	$(LINK) -arch=native  main.o $(OBJECTS) $(LIBS) -o $@
libcup.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@
clean:
	rm -f simulation $(OBJECTS)
	rm -f *.o
