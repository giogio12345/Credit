#ifndef __tools_gpu_cuh
#define __tools_gpu_cuh

#include "cuda.h"
#include "PrecisionTypes.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//! We gather here some functions used in the whole program
namespace tools_gpu
{

//! A simple tool to check errors from kernel and CUDA API functions
void checkCUDAError(const char * msg, const char * file, const unsigned line);

//! A simple tool that returns the cuda capability of the device
unsigned get_cuda_arch();

//! A kernel used by the previous function
__global__
void get_cuda_arch(unsigned *);

//! A function used to free a vector of device pointers
void cudaFree_wrapper(std::vector<number *> & p, unsigned num_gpus);

//! A simple function to get the number of available GPU
unsigned get_device_count();

}


#endif
