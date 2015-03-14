#ifndef __Constants_hpp
#define __Constants_hpp

#include <limits>

//! We gather here some of the constants used in the whole program.
namespace Constants
{
const unsigned max_threads_per_block=1024;
const number alpha=1.95996398454005423552;
const number quantile=0.99;
const number epsilon=std::numeric_limits<number>::epsilon();
}

#endif
