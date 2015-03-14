#ifndef __cdo_cuh
#define __cdo_cuh

#include <stdio.h>

#include <cublas_v2.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/reduce.h>
#include <cassert>

#include "Type.hpp"
#include "MC_Type.hpp"
#include "PrecisionTypes.hpp"
#include "tools.hpp"
#include "tools_gpu.cuh"

//! We gather here the functions which evaluate the fair spread of a Synthetic CDO on multi-GPUs
namespace cdo_gpu
{

//!
/*! A simple __device__ function that returns the global index of a thread.
 */
__device__ unsigned get_index();

//!
/*! A __device__ function that maps time to period.
 */
__device__ unsigned timetoperiod(number t, unsigned maxidx, number daycount);

//!
/*! This kernel does some calculation for the values of the floating and the fixed leg.
 */
__global__ void evaluate_NDef (	number * d_T, number * d_NDef, unsigned aidx, unsigned didx, unsigned maxidx,
                                number daycount, number dfrac, number afrac, number Tmax, number R, number r,
                                number npf, unsigned nsim, unsigned N);

//!
/*! This kernel evaluates the floatin leg
 */
__global__ void evaluate_Vflt (number * d_Vflt, number * d_VVflt, number * d_T, number aidx, number didx, unsigned nsim);

//!
/*! This kernel evaluates the fixed leg
 */
__global__ void evaluate_Vfix(	number * d_Vfix, number * d_VVfix, number * d_T, number * d_NDef, number * d_Notional,
                                number * d_LastNotional, unsigned maxidx, number npf, number R, number r,
                                number daycount, number c, unsigned nsim);

//!
/*! This kernel evaluates the correlation between the floating and the fixed leg
 */
__global__ void evaluate_CVfltVfix(number * d_CVfltVfix, number * d_Vflt, number * d_Vfix, unsigned nsim);

//!
/*! This function evaluates on multi-GPU the fair spread of a Kth-to-Default Swap. This is the argument list:
 * \param d_T2          vector of device pointers che contengono gli indirizzi delle matrici dei tempi di default
 * \param N             number of firms
 * \param R             recovery rate
 * \param npf           notional per firm
 * \param a             attachment point
 * \param d             detachment point
 * \param c             coupon on fixed leg
 * \param r             risk-free interest rate
 * \param T             maturity
 * \param daycount      coupon frequency per year
 * \param isPB          true for ProtectionBuyer, false for ProtectionSeller (just changes the sign of the contract value)
 * \param nsim          number of simulations
 * \param bt            a simple object that handles blocks and threads
 * \param results       a vector where to write the results
 * \param num_gpus      number of available gpus
 */
void evaluate_cdo(std::vector<number *> d_T2, unsigned N, number R, number npf, number a, number d, number c, number r, number T, number daycount, bool isProtectionBuyer, unsigned nsim, tools::Block_Threads & bt, std::vector<number> & results, unsigned num_gpus);

}

#endif
