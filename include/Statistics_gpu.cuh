#ifndef __statistics_gpu_cuh
#define __statistics_gpu_cuh

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Type.hpp"
#include "MC_Type.hpp"
#include "PrecisionTypes.hpp"

//! We gather here the statistic functions on GPU
namespace statistics_gpu
{

//! This function generates a gamma random variable
/*! Code adapted from http://www.johndcook.com/stand_alone_code.html
 *  All code here is in the public domain. Do whatever you want with it, no strings attached.
 */
__device__ number GetGamma(curandState * uniform_state, curandState * normal_state, number shape, number scale);

//! A simple __device__function that generates a uniform random variable
__device__ number get_uniform(curandState * localState);

//! A simple __device__function that generates a gaussian random variable
__device__ number get_normal(curandState * localState);

//! This function generates a chi squared with non-integer degrees of freedom
/*! Code adapted from http://www.johndcook.com/stand_alone_code.html
 *  All code here is in the public domain. Do whatever you want with it, no strings attached.
 */
__device__ number get_chi(number dof, curandState * uniform_state, curandState * normal_state);

}

#endif
