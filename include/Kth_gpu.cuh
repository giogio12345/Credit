#ifndef __kth_gpu_cuh
#define __kth_gpu_cuh

#include <stdio.h>

#include <cuda.h>
#include <vector>
#include <iostream>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/reduce.h>

#include "Type.hpp"
#include "MC_Type.hpp"
#include "PrecisionTypes.hpp"
#include "tools_gpu.cuh"
#include "tools.hpp"

//! We gather here the functions which evaluate the fair spread of a Kth-to-Default Swap on multi-GPUs
namespace kth_gpu
{

//!
/*! A simple __device__ function that returns the global index of a thread.
 */
__device__ unsigned get_index();

//!
/*! This kernel evaluates the Premium Leg of the Swap
 */
__global__ void evaluate_PL(unsigned * d_SP, number * d_T, unsigned n, unsigned Kth, number delta, unsigned dim, unsigned paths);

//!
/*! This kernel evaluates the Premium Leg of the Swap
 */
__global__ void evaluate_DP (number * d_T, number * d_DP, number r, number R, number T, unsigned Kth, unsigned dim, unsigned paths);

//!
/*! This kernel evaluates the Premium Leg of the Swap
 */
__global__ void evaluate_AP (number * d_AP, number * d_T, unsigned Kth, unsigned n, unsigned paths, unsigned dim, number delta, number r);

//!
/*! This function evaluates on multi-GPU the fair spread of a Kth-to-Default Swap. This is the argument list:
 * \param Kth           seniority of the contract
 * \param d_T2          vector of device pointers che contengono gli indirizzi delle matrici dei tempi di default
 * \param N             number of firms
 * \param R             recovery rate
 * \param npf           notional per firm
 * \param r             risk-free interest rate
 * \param T             maturity
 * \param daycount      coupon frequency per year
 * \param paths         number of simulations
 * \param bt            a simple object that handles blocks and threads
 * \param results       a vector where to write the results
 * \param num_gpus      number of available gpus
 */
void evaluate_kth(unsigned Kth, std::vector<number *> d_T2, unsigned N, number R, number npf, number r, number T, number daycount, unsigned paths, tools::Block_Threads & bt, std::vector<number> & results, unsigned num_gpus);

}

#endif
