#ifndef __DefTimes_cuh
#define __DefTimes_cuh

#include <curand.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iomanip>
#include <cassert>

#include "tools_gpu.cuh"
#include "Type.hpp"
#include "MC_Type.hpp"
#include "PrecisionTypes.hpp"
#include "Statistics_gpu.cuh"
#include "tStudent.cuh"
#include "Constants.hpp"
#include "tools.hpp"

//! We gather here the functions which generate the default times on multi-GPUs
namespace deftimes_gpu
{

//!
/*! A simple __device__ function that returns the global index of a thread.
 */
__device__ unsigned get_index();

//!
/*! An heapsort __device__ function, taken from http://en.wikibooks.org/wiki/Algorithm_Implementation/Sorting/Heapsort#In-place_heapsort .
 */
__device__ void sort(number * v, unsigned id, unsigned paths, unsigned n);

//!
/*! This kernel transforms gaussian samples in default times.
 */
__global__ void normal_to_deftimes(number * d_T, number lambda, unsigned paths, unsigned N, number epsilon);
//!
/*! This kernel sums squared gaussian numbers in order to generate chi squared samples.
 */
__global__ void normal_to_chi(number * d_T, number * d_chi, unsigned paths, unsigned dof, number epsilon);

//!
/*! This kernel transforms Student's t samples in default times.
 */
__global__ void t_to_deftimes(number * d_T, number * d_C, number lambda, unsigned paths, unsigned N, unsigned dof, number epsilon);

//!
/*! This kernel sorts the matrix
 */
__global__ void sorting (number * d_T, unsigned paths, unsigned N, number lambda, MC_Type mc_type);

//!
/*! This function is a wrapper for the cuBLAS implementation of the matrix-matrix multiplication.
 */
void gpu_blas_mmul(const number * A, const number * B, number * C, unsigned N, unsigned paths);

//!
/*! This function generates on multi-GPU matrices with default times correlated according to a gaussian copula. This is the argument list:
 * \param mc_type_      type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
 * \param dim           dimension of the reference portfolio
 * \param nsim          number of simulation per GPU
 * \param lambda        hazard rate
 * \param h_S           host pointer to row-major correlation matrix
 * \param bt            a simple object that handles blocks and threads
 * \param num_gpus      number of available gpus
 * \param seed          a seed for the pseudo random generator
 */
std::vector<number *> generate_deftimes_guassian_GPU(MC_Type mc_type, unsigned dim, unsigned nsim, number lambda, number * h_S, tools::Block_Threads & bt, unsigned num_gpus, unsigned seed);

//!
/*! This function generates on multi-GPU matrices with default times correlated according to a gaussian copula. This is the argument list:
 * \param mc_type_      type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
 * \param dim           dimension of the reference portfolio
 * \param nsim          number of simulation per GPU
 * \param lambda        hazard rate
 * \param h_S           host pointer to row-major correlation matrix
 * \param dof           degrees of freedom of the chi squared
 * \param bt            a simple object that handles blocks and threads
 * \param num_gpus      number of available gpus
 * \param seed          a seed for the pseudo random generator
 */
std::vector<number *> generate_deftimes_t_GPU(MC_Type mc_type, unsigned dim, unsigned nsim, number lambda, number * h_S, unsigned dof, tools::Block_Threads & bt, unsigned num_gpus, unsigned seed);

}

#endif
