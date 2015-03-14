# You should add here your NVCC and CXX compilers and a path to Eigen and Boost

NVCC=/cineca/prod/compilers/cuda/6.5.14/none/bin/nvcc
CXX=/cineca/prod/compilers/gnu/4.8.0/none/bin/g++
EIGEN=../../Eigen
BOOST=

##############################################################################################################################

NVCC_ARCH=-gencode arch=compute_35,code=sm_35
NVCCFLAGS	= --compiler-options -Wall --compiler-options -Wno-switch --compiler-options -Wno-unused-local-typedefs --compiler-options -Wno-unused-function --compiler-options -Wsign-compare --compiler-options -Wpointer-arith -m64 -maxrregcount=64 #--ptxas-options=-v
CXXLDFLAGS	= -lcudart -lcublas -lcurand
INCLUDES= -Iinclude -I$(EIGEN) -I$(BOOST)
CXX_RELEASE_FLAGS= -O2 -funroll-loops -funroll-all-loops -fstrict-aliasing
CXX_FLAGS= -pedantic -Wall -Wpointer-arith -Wsign-compare -Wno-long-long -Wno-switch
NVCC_RELEASE_FLAGS= -O2 --use_fast_math --compiler-options -O2 --compiler-options -funroll-loops --compiler-options -funroll-all-loops --compiler-options -fstrict-aliasing 
DEBUG_FLAGS=-g -G -O0
NVCC_OMP= --compiler-options -fopenmp 

OBJ_CPU = obj/DefTimesGen.o obj/tools.o obj/Statistics.o
OBJ_GPU = obj/Statistics_gpu.o obj/DefTimes.o obj/Cdo_gpu.o obj/tools_gpu.o obj/Kth_gpu.o obj/Cva_gpu.o
OBJ_EXE= obj/test1.o obj/test2.o obj/test3.o obj/test4.o obj/test5.o obj/test6.o obj/test7.o obj/test8.o 

NVCC_FLAGS	= -ccbin=$(CXX) $(NVCC_OMP) $(NVCCFLAGS) $(NVCC_ARCH) $(INCLUDES) $(NVCC_RELEASE_FLAGS)
NVCC_LD_FLAGS	= -ccbin=$(CXX) $(NVCC_OMP) $(CXXLDFLAGS) $(NVCC_ARCH)

TEST_EXE=test1 test2 test3 test4 test5 test6 test7 test8

all: $(OBJ_CPU) $(OBJ_GPU)
	ar r lib/libcredit.a $(OBJ_CPU) $(OBJ_GPU)
	ranlib lib/libcredit.a
	
test: $(OBJ_EXE) 
	$(NVCC) $(NVCC_LD_FLAGS) -rdc=true -Llib -lcredit obj/test1.o -o test1
	$(NVCC) $(NVCC_LD_FLAGS) -rdc=true -Llib -lcredit obj/test2.o -o test2
	$(NVCC) $(NVCC_LD_FLAGS) -rdc=true -Llib -lcredit obj/test3.o -o test3
	$(NVCC) $(NVCC_LD_FLAGS) -rdc=true -Llib -lcredit obj/test4.o -o test4
	$(NVCC) $(NVCC_LD_FLAGS) -rdc=true -Llib -lcredit obj/test5.o -o test5
	$(NVCC) $(NVCC_LD_FLAGS) -rdc=true -Llib -lcredit obj/test6.o -o test6
	$(NVCC) $(NVCC_LD_FLAGS) -rdc=true -Llib -lcredit obj/test7.o -o test7
	$(NVCC) $(NVCC_LD_FLAGS) -rdc=true -Llib -lcredit obj/test8.o -o test8
		
obj/%.o: src/%.cpp
	$(NVCC) $(NVCC_FLAGS) --compiler-options -std=c++11 -c -o $@ $<
	
obj/%.o: src/%.cu
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $<

clean:
	$(RM) obj/*.o lib/libcredit.a

distclean: clean
	$(RM) $(TEST_EXE)
