#ifndef __PrecisionTypes_
#define __PrecisionTypes_

#include <omp.h>

#define __CUDA_ERROR_CHECK__

//#define __SINGLE_PRECISION__

#ifdef __SINGLE_PRECISION__
//! A simple typedef for float
typedef float number;
#else
//! A simple typedef for double
typedef double number;
#endif

#endif
