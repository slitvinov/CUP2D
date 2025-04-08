.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp .cu .o

CXX = g++
CXXFLAGS = -O2 -g
NVCC = nvcc
LINK = $(NVCC)
LIBS = -lcublas -lcusparse
OPENMPFLAGS = -fopenmp

FLAGS =\
-D_BS_=8\
$(OPENMPFLAGS)\

O =\
cuda.o\
main.o\

main: $O
	$(LINK) -o main $O $(LDFLAGS) -Xcompiler '$(OPENMPFLAGS)' $(LIBS)
.cpp.o:
	$(CXX) -c $< $(FLAGS) $(CXXFLAGS)
.cu.o:
	$(NVCC) -c $< $(NVCCFLAGS)
clean:
	rm -f main $O

main.o: utils.h
