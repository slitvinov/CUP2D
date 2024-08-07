config ?= production
precision ?= double
bs ?= 8
gpu ?= false
openmp ?= false
onetbb ?= false
symmetry ?= false
cylinder_ref ?= false

# SET FLAGS FOR COMPILER
ifneq ($(MPICXX),)
	CXX=$(MPICXX)
else
	CXX=mpic++
endif
LINK = $(CXX)

# LOAD FLAGS -- GCC NEEDED
CPPFLAGS+= -std=c++17 -g

ifeq "$(openmp)" "true"
	CPPFLAGS+= -fopenmp
endif

# FLAGS FOR EXTERNAL LIBRARIES
LIBS+= -lgsl -lgslcblas -lhdf5

# ADD LIBRARY PATHS IF GIVEN
ifneq ($(HDF5_ROOT),)
	LIBS     += -L$(HDF5_ROOT)/lib
	CPPFLAGS += -I$(HDF5_ROOT)/include
endif

ifneq ($(GSL_ROOT),)
	CPPFLAGS += -I$(GSL_ROOT)/include
	LIBS += -L$(GSL_ROOT)/lib
endif

#################################################
# oneTBB
#################################################
ifeq "$(onetbb)" "true"
	LIBS     += -L$(ONETBBROOT)/lib64 -ltbb
	CPPFLAGS += -I$(ONETBBROOT)/include
	CPPFLAGS += -DCUBISM_USE_ONETBB
endif

# ENABLE OPTIMIZATION/DEBUG FLAGS IF WISHED
ifeq "$(findstring prod,$(config))" ""
	CPPFLAGS+= -O0
	ifeq "$(config)" "segf"
		CPPFLAGS+= -fsanitize=address
		LIBS+= -fsanitize=address -static-libasan
	endif
	ifeq "$(config)" "nans"
		CPPFLAGS+= -fsanitize=undefined
		LIBS+= -fsanitize=undefined
	endif
else
	CPPFLAGS+= -DNDEBUG -O3 -fstrict-aliasing -march=native -mtune=native
endif

# SET FLOATING POINT ACCURACY
ifeq "$(precision)" "single"
	CPPFLAGS += -D_FLOAT_PRECISION_
else ifeq "$(precision)" "double"
	CPPFLAGS += -D_DOUBLE_PRECISION_
else ifeq "$(precision)" "long_double"
	CPPFLAGS += -D_LONG_DOUBLE_PRECISION_
endif

# SET VPATH FOR MAKE TO SEARCH FOR FILES
BUILDDIR = .
DIRS = $(sort $(dir $(wildcard ../source/*) $(wildcard ../source/*/)))
VPATH := $(DIRS) $(BUILDDIR)/../Cubism/src/

ifeq "$(symmetry)" "true"
	CPPFLAGS += -DCUP2D_PRESERVE_SYMMETRY
endif

ifeq "$(cylinder_ref)" "true"
	CPPFLAGS += -DCUP2D_CYLINDER_REF
endif

# SET FLAGS FOR CUBISM
CPPFLAGS+= -D_BS_=$(bs) -DCUBISM_ALIGNMENT=32
CPPFLAGS += -I.. -DDIMENSION=2

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

OBJECTS = $(S:.cpp=.o)

#################################################
# CUDA
#################################################
NVCC ?= nvcc
OBJECTS += ExpAMRSolver.o BiCGSTAB.o LocalSpMatDnVec.o
CPPFLAGS += -fopenmp -DGPU_POISSON
NVCCFLAGS += -arch=native -std=c++17 -O3 --use_fast_math -Xcompiler "$(CPPFLAGS)" -DGPU_POISSON
LIBS += -lcublas -lcusparse

# DEFINE COMPILATION TARGETS
all: simulation libcup.a
.DEFAULT: all

# COMPILATION INSTRUCTIONS FOR APPLICATION AND LIBRARY
simulation: source/main.o $(OBJECTS)
	$(LINK) -arch=native  main.o $(OBJECTS) $(LIBS) -o $@
libcup.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

# COMPILATION INSTRUCTIONS FOR OBJECT FILES
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

# COMPILATION INSTRUCTION FOR CLEANING BUILD
clean:
	rm -f simulation libcup.a
	rm -f *.o
