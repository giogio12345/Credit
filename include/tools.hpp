#ifndef __tools_hpp
#define __tools_hpp

#include <cmath>
#include <iostream>

#include "PrecisionTypes.hpp"
#include "Constants.hpp"
#include "tools_gpu.cuh"

//! We gather here some objects used in the whole program
namespace tools
{

//! A simple class used to handle blocks and threads in the launch of CUDA kernels.
class Block_Threads
{
private:
        unsigned max_threads;
        unsigned cuda_arch;
public:
        Block_Threads();
        Block_Threads(unsigned);
        void set_max_thread_per_block(unsigned);
        std::pair<unsigned, unsigned> evaluate_bt(unsigned);
};

//! A simple class used as a template-independent counter for Option-type objects.
class Counter
{
public:
        static unsigned & counter()
        {
                static unsigned counter_=0;
                return counter_;
        }
};

}

#endif