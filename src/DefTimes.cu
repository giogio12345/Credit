#include "DefTimes.cuh"
#include "tStudent.cu"

namespace deftimes_gpu
{

__device__
unsigned get_index()
{
        unsigned index_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned index_y = blockIdx.y * blockDim.y + threadIdx.y;

        unsigned grid_width = gridDim.x * blockDim.x;
        unsigned index = index_y * grid_width + index_x;

        return index;
}

__device__ void sort(number * v, unsigned id, unsigned paths, unsigned N)
{
        unsigned n = N;
        unsigned i = n/2;
        unsigned parent;
        unsigned child;
        bool found;
        bool sorted=false;
        number t;

        while (!sorted)   // Loops until arr is sorted
        {
                if (i > 0)   // First stage - Sorting the heap
                {
                        --i;           // Save its index to i
                        t = v[id+i*paths];    // Save parent value to t
                }
                else         // Second stage - Extracting elements in-place
                {
                        --n;           // Make the new heap smaller
                        if (n == 0)
                        {
                                sorted=true;        // When the heap is empty, we are done
                        }
                        t = v[id+n*paths];    // Save last value (it will be overwritten)
                        v[id+n*paths] = v[id]; // Save largest value at the end of arr
                }

                parent = i; // We will start pushing down t from parent
                child = i*2 + 1; // parent's left child

                // Sift operation - pushing the value of t down the heap
                found=false;
                while (child < n && !found)
                {
                        if (child + 1 < n  &&  v[id+(child+1)*paths] > v[id+child*paths])
                        {
                                ++child; // Choose the largest child
                        }
                        if (v[id+child*paths] > t)   // If any child is bigger than the parent
                        {
                                v[id+parent*paths] = v[id+child*paths]; // Move the largest child up
                                parent = child; // Move parent pointer to this child
                                child = parent*2+1; // the previous line is wrong
                        }
                        else
                        {
                                found=true; // t's place is found
                        }
                }
                v[id+parent*paths] = t; // We save t in the heap
        }
}

__global__ void normal_to_deftimes(number * d_T, number lambda, unsigned paths, unsigned N, number epsilon)
{
        unsigned id=get_index();
        if (id<paths*N)
        {
                number temp=-log(normcdf(d_T[id]))/lambda;
                if (fabs(temp)<epsilon)
                {
                        temp+=epsilon;
                }
                d_T[id]=temp;
        }
}

__global__ void normal_to_chi(number * d_T, number * d_chi, unsigned paths, unsigned dof, number epsilon)
{
        unsigned id=get_index();
        if (id<paths)
        {
                number temp=0;
                number chi=0;
                for (unsigned i=0; i<dof; ++i)
                {
                        temp=d_T[i*paths+id];
                        chi+=temp*temp;
                }
                if (temp<epsilon)
                {
                        chi+=epsilon;
                }
                d_chi[id]=chi;
        }
}

// in d_T ho le chi, in d_C ho le normali, d_C resta
__global__ void t_to_deftimes(number * d_T, number * d_C, number lambda, unsigned paths, unsigned N, unsigned dof, number epsilon)
{
        unsigned id=get_index();
        if (id<paths*N)
        {
                number temp=d_C[id]*sqrtf(dof/d_T[id%paths]);
                temp=-log(tnc(temp, dof, 0.))/lambda;
                if (fabs(temp)<epsilon)
                {
                        temp+=epsilon;
                }
                d_C[id]=temp;
        }
}

__global__ void sorting (number * d_T, unsigned paths, unsigned N, number lambda, MC_Type mc_type)
{
        unsigned id=get_index();

        // just a trick to avoid an issue with CURAND normal Sobol generator which return nan at high dimension
        if (id==0 && mc_type==QMC)
        {
                for (unsigned i=0; i<N; ++i)
                {
                        d_T[i*paths]=8.29236/lambda;
                }
        }
        else if (id<paths)
        {
                sort(d_T, id, paths, N);
        }

}

void gpu_blas_mmul(const number * A, const number * B, number * C, unsigned N, unsigned paths)
{

        const number alf = 1;
        const number bet = 0;
        const number * alpha = &alf;
        const number * beta = &bet;


        // Create a handle for CUBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Do the actual multiplication
#ifdef __SINGLE_PRECISION__
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, paths, N, N, alpha, B, paths, A, N, beta, C, paths);
#ifdef __CUDA_ERROR_CHECK__
        cudaDeviceSynchronize();
        tools_gpu::checkCUDAError("cublasSgemm", __FILE__, __LINE__);
#endif

#else
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, paths, N, N, alpha, B, paths, A, N, beta, C, paths);
#ifdef __CUDA_ERROR_CHECK__
        cudaDeviceSynchronize();
        tools_gpu::checkCUDAError("cublasDgemm", __FILE__, __LINE__);
#endif

#endif

        // Destroy the handle
        cublasDestroy(handle);
}

std::vector<number *> generate_deftimes_guassian_GPU(MC_Type mc_type, unsigned dim, unsigned nsim, number lambda, number * h_S, tools::Block_Threads & bt, unsigned num_gpus, unsigned seed)
{
        std::vector<number *> d_times(num_gpus, NULL);
        omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices

        #pragma omp parallel
        {
                std::pair<unsigned, unsigned> b_t(0,0);

                unsigned cpu_thread_id = omp_get_thread_num();
                cudaSetDevice(cpu_thread_id);

                //nsim=nsim/num_gpus;

                // Generating random numbers
                thrust::device_vector<number> S(dim*dim);
                thrust::device_vector<number> times(nsim*dim);

                number * d_S=raw_pointer_cast(&S[0]);
                number * d_T=raw_pointer_cast(&times[0]);

                if (mc_type==MC)
                {
                        curandGenerator_t gen;
                        // Create pseudo-random number generator
                        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandCreateGenerator", __FILE__, __LINE__);
#endif
                        // Set seed
                        curandSetPseudoRandomGeneratorSeed(gen, seed);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandSetPseudoRandomGeneratorSeed", __FILE__, __LINE__);
#endif
                        // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                        curandGenerateNormal(gen, d_T, nsim*dim, 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandGenerateNormal", __FILE__, __LINE__);
#endif

#else
                        curandGenerateNormalDouble(gen, d_T, nsim*dim, 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandGenerateNormalDouble", __FILE__, __LINE__);
#endif

#endif
                        curandDestroyGenerator(gen);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandDestroyGenerator", __FILE__, __LINE__);
#endif
                }
                // curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions);
                else
                {
                        curandGenerator_t gen;
                        // Create pseudo-random number generator
                        curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandCreateGenerator", __FILE__, __LINE__);
#endif
                        // Set seed
                        curandSetQuasiRandomGeneratorDimensions(gen, dim);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandSetQuasiRandomGeneratorDimensions", __FILE__, __LINE__);
#endif
                        // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                        curandGenerateNormal(gen, d_T, nsim*dim, 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandGenerateNormal", __FILE__, __LINE__);
#endif

#else
                        curandGenerateNormalDouble(gen, d_T, nsim*dim, 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandGenerateNormalDouble", __FILE__, __LINE__);
#endif

#endif
                        curandDestroyGenerator(gen);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandDestroyGenerator", __FILE__, __LINE__);
#endif
                }

                // Copying chol decompisition and calculating correlated random numbers
                cudaMemcpy(d_S,h_S,dim*dim*sizeof(number),cudaMemcpyHostToDevice);
#ifdef __CUDA_ERROR_CHECK__
                tools_gpu::checkCUDAError("cudaMemcpy", __FILE__, __LINE__);
#endif

                number * d_C;
                cudaMalloc(&d_C, nsim * dim * sizeof(number));
#ifdef __CUDA_ERROR_CHECK__
                tools_gpu::checkCUDAError("cudaMalloc", __FILE__, __LINE__);
#endif

                gpu_blas_mmul(d_S, d_T, d_C, dim, nsim);

                d_T=d_C;

                // Calculating default times
                b_t=bt.evaluate_bt(nsim*dim);

                normal_to_deftimes<<<b_t.second, b_t.first>>> (d_T, lambda, nsim, dim, Constants::epsilon);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                // Sorting default times
                b_t=bt.evaluate_bt(nsim);

                sorting<<<b_t.second, b_t.first>>> (d_T, nsim, dim, lambda, mc_type);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                d_times[cpu_thread_id]=d_T;

                cudaDeviceSynchronize();
        }

        return d_times;
}

std::vector<number *> generate_deftimes_t_GPU(MC_Type mc_type, unsigned dim, unsigned nsim, number lambda, number * h_S, unsigned dof, tools::Block_Threads & bt, unsigned num_gpus, unsigned seed)
{
        std::vector<number *> d_times(num_gpus, NULL);
        omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices

        std::pair<unsigned, unsigned> b_t(0,0);

        #pragma omp parallel
        {
                unsigned cpu_thread_id = omp_get_thread_num();
                cudaSetDevice(cpu_thread_id);

                // Generating random numbers
                thrust::device_vector<number> S(dim*dim);
                thrust::device_vector<number> times(nsim*dim);
                thrust::device_vector<number> chi(nsim, 0.);

                number * d_S=raw_pointer_cast(&S[0]);
                number * d_T=raw_pointer_cast(&times[0]);
                number * d_chi=raw_pointer_cast(&chi[0]);

                {
                        thrust::device_vector<number> data(nsim*(dim+dof));
                        number * d_data=raw_pointer_cast(&data[0]);

                        if (mc_type==MC)
                        {
                                curandGenerator_t gen;
                                // Create pseudo-random number generator
                                curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandCreateGenerator", __FILE__, __LINE__);
#endif
                                // Set seed
                                curandSetPseudoRandomGeneratorSeed(gen, seed);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandSetPseudoRandomGeneratorSeed", __FILE__, __LINE__);
#endif
                                // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                                curandGenerateNormal(gen, d_data, nsim*(dim+dof), 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandGenerateNormal", __FILE__, __LINE__);
#endif

#else
                                curandGenerateNormalDouble(gen, d_data, nsim*(dim+dof), 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandGenerateNormalDouble", __FILE__, __LINE__);
#endif

#endif
                                curandDestroyGenerator(gen);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandDestroyGenerator", __FILE__, __LINE__);
#endif
                        }
                        // curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions);
                        else
                        {
                                curandGenerator_t gen;
                                // Create pseudo-random number generator
                                curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandCreateGenerator", __FILE__, __LINE__);
#endif
                                // Set seed
                                curandSetQuasiRandomGeneratorDimensions(gen, (dim+dof));
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandSetQuasiRandomGeneratorDimensions", __FILE__, __LINE__);
#endif
                                // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                                curandGenerateNormal(gen, d_data, nsim*(dim+dof), 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandGenerateNormal", __FILE__, __LINE__);
#endif

#else
                                curandGenerateNormalDouble(gen, d_data, nsim*(dim+dof), 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandGenerateNormalDouble", __FILE__, __LINE__);
#endif

#endif
                                curandDestroyGenerator(gen);
#ifdef __CUDA_ERROR_CHECK__
                                tools_gpu::checkCUDAError("curandDestroyGenerator", __FILE__, __LINE__);
#endif
                        }

                        cudaMemcpy(d_T, d_data, nsim*dim*sizeof(number), cudaMemcpyDeviceToDevice);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("cudaMemcpy", __FILE__, __LINE__);
#endif
                        b_t=bt.evaluate_bt(nsim);

                        normal_to_chi<<<b_t.second, b_t.first>>>(&d_data[nsim*dim], d_chi, nsim, dof, Constants::epsilon);
#ifdef __CUDA_ERROR_CHECK__
                        cudaDeviceSynchronize();
                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                }

                // Copying chol decompisition and calculating correlated random numbers
                cudaMemcpy(d_S, h_S, dim*dim*sizeof(number), cudaMemcpyHostToDevice);
#ifdef __CUDA_ERROR_CHECK__
                tools_gpu::checkCUDAError("cudaMemcpy", __FILE__, __LINE__);
#endif

                number * d_C;
                cudaMalloc(&d_C, nsim*dim*sizeof(number));
#ifdef __CUDA_ERROR_CHECK__
                tools_gpu::checkCUDAError("cudaMalloc", __FILE__, __LINE__);
#endif

                gpu_blas_mmul(d_S, d_T, d_C, dim, nsim);

                // Calculating default times
                // in d_T ho le chi, in d_C ho le normali, d_C resta
                b_t=bt.evaluate_bt(nsim*dim);

                t_to_deftimes<<<b_t.second, b_t.first>>> (d_chi, d_C, lambda, nsim, dim, dof, Constants::epsilon);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                // Sorting default times
                b_t=bt.evaluate_bt(nsim);

                sorting<<<b_t.second, b_t.first>>> (d_C, nsim, dim, lambda, mc_type);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                d_times[cpu_thread_id]=d_C;

                cudaDeviceSynchronize();

        }

        return d_times;
}

}
