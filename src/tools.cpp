#include "tools.hpp"

namespace tools
{

Block_Threads::Block_Threads()
{
        max_threads=Constants::max_threads_per_block;
        cuda_arch=tools_gpu::get_cuda_arch();
        if (max_threads>1024)
        {
                throw(std::logic_error("Number of threads too large (max 1024)."));
        }
}
Block_Threads::Block_Threads(unsigned max_threads_per_block)
{
        max_threads=max_threads_per_block;
        cuda_arch=tools_gpu::get_cuda_arch();
        if (max_threads>1024)
        {
                throw(std::logic_error("Number of threads too large (max 1024)."));
        }
}
void Block_Threads::set_max_thread_per_block(unsigned max_threads_per_block)
{
        max_threads=max_threads_per_block;
        if (max_threads>1024)
        {
                throw(std::logic_error("Number of threads too large (max 1024)."));
        }
}
std::pair<unsigned, unsigned> Block_Threads::evaluate_bt(unsigned n)
{
        unsigned threads, blocks;
        if (n<max_threads)
        {
                threads=std::pow(2, ceil(log(n)/log(2)));
                blocks=1;
        }
        else
        {
                threads=max_threads;
                blocks=n/max_threads+1;
                if (blocks>65536 && cuda_arch<300)
                {
                        throw(std::logic_error("Number of blocks too large for Fermi devices."));
                }
        }
        std::pair<unsigned, unsigned> bt(threads, blocks);
        return bt;
}


}
