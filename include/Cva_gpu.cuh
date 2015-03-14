#ifndef __cva_cuh
#define __cva_cuh

#include <stdio.h>

#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/count.h>

#include "Type.hpp"
#include "MC_Type.hpp"
#include "PrecisionTypes.hpp"
#include "RateModels.hpp"
#include "Statistics_gpu.cuh"
#include "tools.hpp"
#include "tools_gpu.cuh"

//! We gather here the functions which evaluate the Expected Exposure of an Interest Rate Swap on multi-GPUs
namespace cva_gpu
{
//!
/*! A simple __device__ function that returns the global index of a thread.
 */
__device__ unsigned get_index();

//!
/*! This function sets up the state of a cuRAND generator.
 */
__global__ void setup_curand(curandState * state, unsigned long seed, unsigned nsim);

//!
/*! This function generates the first scenarios of the Vasicek and Hull-White rate models.
 */
__global__ void generate_rates_0(number * d_scen, unsigned nsim, number spot_rate, number mean_mult, number mean_add, number sq_var);

//!
/*! This function generates the scenarios of the Vasicek and Hull-White rate models.
 */
__global__ void generate_rates_i(number * d_scen, number * d_scen_old, unsigned nsim, number spot_rate, number mean_mult, number mean_add, number sq_var);

//!
/*! This function generates the first scenarios of the CIR rate model.
 */
__global__ void generate_rates_0_CIR(number * d_scen, unsigned nsim, number c, number l, number d, curandState * u, curandState * n);

//!
/*! This function generates the scenarios of the CIR rate model.
 */
__global__ void generate_rates_i_CIR(number * d_scen, number * d_scen_old, unsigned nsim, number c, number e, number d, curandState * u, curandState * n);

//!
/*! This function evaluates the Discount Factor og the Hull-White and CIR models.
 */
__global__ void evaluate_DF(number * d_DF, number * d_A, number * d_B, number * d_scen, unsigned pricing_grid_size, unsigned nsim, unsigned i);

//!
/*! This function evaluates the Discount Factor og the Vasicek model.
 */
__global__ void evaluate_DF_V(number * d_DF, number * d_A, number * d_B, number * d_scen, unsigned pricing_grid_size, unsigned nsim, unsigned i);

//!
/*! This function evaluates the Mark-to-Market of the Interest Rate Swap
 */
__global__ void evaluate_MtM(number * d_MtM, number * d_DF, number * d_irsCF, unsigned i, unsigned nsim, unsigned pricing_grid_size, unsigned simulation_grid);

//!
/*! This function evaluates the Positive and Negative Potential Future Exposure of the Interest Rate Swap
 */
__global__ void evaluate_PFE(number * d_PFE, number * d_NPFE, number * d_MtM, unsigned nsim, number alpha);

//! A simple class used as a policy in the thrust function transform_reduce
/*! This class is passed to the thrust function transform_reduce and allows us to sum only over positive values.
 */
template<typename T>
struct positive_value: public thrust::unary_function<T,T>
{
        __host__ __device__ T operator()(const T & x) const
        {
                return x>0. ? x  : 0.;
        }
};

//! A simple class used as a policy in the thrust function transform_reduce
/*! This class is passed to the thrust function transform_reduce and allows us to sum only over negative values.
 */
template<typename T>
struct negative_value: public thrust::unary_function<T,T>
{
        __host__ __device__ T operator()(const T & x) const
        {
                return x<0. ? x  : 0.;
        }
};

//!
/*! This function evaluates the Expected Exposure of an Interest Rate Swap (the actual calculation of the CVA is made on the CPU). The argument list is the following:
 * \param rate_model            RateModels type: Vasicek, HullWhite, CIR
 * \param mc_type               type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
 * \param nsim                  number of simulation
 * \param simulation_grid       instants of simulation
 * \param A_                    matrix of the A values
 * \param B_                    matrix of the B values
 * \param pricing_grid          vector of the pricing grid
 * \param data                  parameters of the various models
 * \param zero_coupon           zero coupon values
 * \param irs_CF                coupon of the swap
 * \param bt                    a simple object that handles blocks and threads
 * \param num_gpus              number of available gpus
 * \param EPE                   vector of the Expected Potential Exposure
 * \param NEPE                  vector of the Negative Expected Potential Exposure
 * \param EE                    vector of the Expected Exposure
 * \param NEE                   vector of the Negative Expected Exposure
 * \param PFE                   vector of the Potential Future Exposure
 * \param NFPE                  vector of the Negative Potential Future Exposure
 * \param seed                  a seed for the pseudo-random generators.
 * \note Quasi Monte Carlo method is not enabled for the CIR simulation.
 */
void evaluate_cva(RateModels rate_model, MC_Type mc_type, unsigned nsim, unsigned simulation_grid, std::vector<number> const & A_, std::vector<number> const & B_, std::vector<number> & pricing_grid, std::vector<number> const & data,  std::vector<std::pair<number, number> > const & zero_coupon, std::vector<number> const & irs_CF, tools::Block_Threads & bt, unsigned num_gpus, number & EPE, number & NEPE, std::vector<number> & EE, std::vector<number> & NEE, std::vector<number> & PFE, std::vector<number> & NPFE, unsigned seed);

}


#endif
