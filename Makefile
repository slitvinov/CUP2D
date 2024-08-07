.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp .cu .o

NVCC = nvcc
MPICXX = mpic++
LINK = nvcc -ccbin='$(MPICXX)'

FLAGS =\
-D_BS_=8\
-DCUBISM_ALIGNMENT=32\
-DDIMENSION=2\
-D_DOUBLE_PRECISION_\
-DGPU_POISSON\
-DNDEBUG\
-I.\
-O3\
-std=c++17\

O =\
source/ArgumentParser.o\
source/main.o\
source/Obstacles/CarlingFish.o\
source/Obstacles/CStartFish.o\
source/Obstacles/CylinderNozzle.o\
source/Obstacles/ExperimentFish.o\
source/Obstacles/FishData.o\
source/Obstacles/Fish.o\
source/Obstacles/Naca.o\
source/Obstacles/NeuroKinematicFish.o\
source/Obstacles/ShapeLibrary.o\
source/Obstacles/ShapesSimple.o\
source/Obstacles/SmartCylinder.o\
source/Obstacles/SmartNaca.o\
source/Obstacles/StefanFish.o\
source/Obstacles/Teardrop.o\
source/Obstacles/Waterturbine.o\
source/Obstacles/Windmill.o\
source/Obstacles/ZebraFish.o\
source/Operators/AdaptTheMesh.o\
source/Operators/advDiff.o\
source/Operators/advDiffSGS.o\
source/Operators/ComputeForces.o\
source/Operators/Forcing.o\
source/Operators/Helpers.o\
source/Operators/PressureSingle.o\
source/Operators/PutObjectsOnGrid.o\
source/Poisson/AMRSolver.o\
source/Poisson/Base.o\
source/Poisson/BiCGSTAB.o\
source/Poisson/ExpAMRSolver.o\
source/Poisson/LocalSpMatDnVec.o\
source/Shape.o\
source/SimulationData.o\
source/Simulation.o\
source/Utils/BufferedLogger.o\

NVCCFLAGS =
main: $O
	$(LINK) -o main $O $(LDFLAGS) -Xcompiler -fopenmp -lcublas -lcusparse -lhdf5 -lgsl -lgslcblas
.cpp.o:
	$(MPICXX) -o $@ -c $< $(FLAGS) $(CXXFLAGS)
.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS) -arch=native -Xcompiler '$(FLAGS)'
clean:
	rm -f main $O
