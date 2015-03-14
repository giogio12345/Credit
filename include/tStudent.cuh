#ifndef __tStudent_gpu_cuh
#define __tStudent_gpu_cuh

#include <cuda.h>
#include "PrecisionTypes.hpp"

// asa243
__device__ number betain ( number x, number p, number q, number beta/*, int * ifault */);
__device__ number tnc ( number t, number df, number delta/*, int * ifault */);

#endif