#include "tools_gpu.cuh"
#include <string>
#include <stdexcept>
#include <boost/lexical_cast.hpp>

namespace tools_gpu
{

void checkCUDAError(const char * msg, const char * file, const unsigned line)
{
        cudaError_t err = cudaGetLastError();
        if( err != cudaSuccess)
        {
                std::string s("Cuda error: ");
                s.append(msg);
                s.append(": ");
                s.append(cudaGetErrorString(err));
                s.append(". In file: ");
                s.append(file);
                s.append(", line: ");
                s.append(boost::lexical_cast<std::string>(line));
                s.append(".");
                throw(std::logic_error(s.c_str()));
                //printf( "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
                //exit(-1);
        }
}

unsigned get_cuda_arch()
{
        thrust::device_vector<unsigned> v(1);

        unsigned * d_v=raw_pointer_cast(&v[0]);

        get_cuda_arch<<<1,1>>> (d_v);
        cudaDeviceSynchronize();
        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);

        thrust::host_vector<unsigned> w(v);

        return w[0];
}

__global__
void get_cuda_arch(unsigned * v)
{
#ifdef __CUDA_ARCH__
        *v=__CUDA_ARCH__;
#endif
}

void cudaFree_wrapper(std::vector<number *> & p, unsigned num_gpus)
{
        omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
        #pragma omp parallel
        {
                unsigned cpu_thread_id = omp_get_thread_num();
                cudaSetDevice(cpu_thread_id);

                cudaFree(p[cpu_thread_id]);
                tools_gpu::checkCUDAError("cudaFree", __FILE__, __LINE__);
        }
}

unsigned get_device_count()
{
        int num_gpus = 0;
        // determine the number of CUDA capable GPUs
        cudaGetDeviceCount(&num_gpus);
        return static_cast<unsigned>(num_gpus);
}

}

