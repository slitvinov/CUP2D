config ?= production
precision ?= double
bs ?= 8
gpu ?= false
openmp ?= false
profile ?= false
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
	#LIBS     += -L/users/chatzima/my_installed_onetbb/lib64 -ltbb
	#CPPFLAGS += -I/users/chatzima/my_installed_onetbb/include
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
CPPFLAGS += -I$(BUILDDIR)/../Cubism/include/ -DDIMENSION=2

OBJECTS = \
		Simulation.o SimulationData.o BufferedLogger.o Helpers.o ArgumentParser.o \
		PressureSingle.o PutObjectsOnGrid.o advDiff.o ComputeForces.o\
		AdaptTheMesh.o AMRSolver.o Shape.o ShapeLibrary.o ShapesSimple.o \
		Fish.o FishData.o SmartCylinder.o StefanFish.o CarlingFish.o  \
		Naca.o CStartFish.o ZebraFish.o NeuroKinematicFish.o  Windmill.o \
		Waterturbine.o Teardrop.o ExperimentFish.o Base.o Forcing.o advDiffSGS.o CylinderNozzle.o \
		SmartNaca.o

#################################################
# CUDA
#################################################
NVCC ?= nvcc
ifeq ("$(gpu)", "true")
	OBJECTS += ExpAMRSolver.o BiCGSTAB.o LocalSpMatDnVec.o
	CPPFLAGS += -fopenmp -DGPU_POISSON
	NVCCFLAGS += -std=c++17 -O3 --use_fast_math -Xcompiler "$(CPPFLAGS)" -DGPU_POISSON
	ifeq ("$(profile)", "true")
		NVCCFLAGS += -DBICGSTAB_PROFILER
	endif
endif

# DEFINE COMPILATION TARGETS
all: debugRL simulation libcup.a
.DEFAULT: all

# COMPILATION INSTRUCTIONS FOR APPLICATION THAT CAN REPRODUCE AN RL RUN
debugRL: debugRL.o $(OBJECTS)
	$(LINK) debugRL.o $(OBJECTS) $(LIBS) -o $@

# COMPILATION INSTRUCTIONS FOR APPLICATION AND LIBRARY
simulation: main.o $(OBJECTS)
	$(LINK) main.o $(OBJECTS) $(LIBS) -o $@
libcup.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

# COMPILATION INSTRUCTIONS FOR OBJECT FILES
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

# COMPILATION INSTRUCTION FOR CLEANING BUILD
clean:
	rm -f debugRL simulation libcup.a
	rm -f *.o
