#ifndef __statistics_gpu2_cu
#define __statistics_gpu2_cu

#include "Statistics_gpu.cuh"

namespace statistics_gpu
{

__device__ number GetGamma(curandState * uniform_state, curandState * normal_state, number shape, number scale)
{
        // Implementation based on "A Simple Method for Generating Gamma Variables"
        // by George Marsaglia and Wai Wan Tsang.  ACM Transactions on Mathematical Software
        // Vol 26, No 3, September 2000, pages 363-372.

        number d, c, x, xsquared, v, u;

        d = shape - 1.0/3.0;
        c = 1.0/sqrt(9.0*d);
        for (;;)
        {
                do
                {
                        x = get_normal(normal_state);
                        v = 1.0 + c*x;
                }
                while (v <= 0.0);
                v = v*v*v;
                u = get_uniform(uniform_state);
                xsquared = x*x;
                if (u < 1.0 -.0331*xsquared*xsquared || log(u) < 0.5*xsquared + d*(1.0 - v + log(v)))
                {
                        return scale*d*v;
                }
        }
}

__device__ number get_uniform(curandState * localState)
{
#ifdef __SINGLE_PRECISION__
        return curand_uniform(localState);
#else
        return curand_uniform_double(localState);
#endif
}

__device__ number get_normal(curandState * localState)
{
#ifdef __SINGLE_PRECISION__
        return curand_normal(localState);
#else
        return curand_normal_double(localState);
#endif
}

__device__ number get_chi(number dof, curandState * uniform_state, curandState * normal_state)
{
        return GetGamma(uniform_state, normal_state, 0.5 * dof, 2.0);
}

}

#endif
