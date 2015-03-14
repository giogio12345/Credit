#include "Cva_gpu.cuh"

namespace cva_gpu
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

__global__ void setup_curand(curandState * state, unsigned long seed, unsigned nsim)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                curand_init ( seed, id, 0, &state[id] );
        }
}

__global__
void generate_rates_0(number * d_scen, unsigned nsim, number spot_rate, number mean_mult, number mean_add, number sq_var)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                d_scen[id]=spot_rate*mean_mult+mean_add+sq_var*d_scen[id];
        }
}

__global__
void generate_rates_i(number * d_scen, number * d_scen_old, unsigned nsim, number spot_rate, number mean_mult, number mean_add, number sq_var)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                d_scen[id]=d_scen_old[id]*mean_mult+mean_add+sq_var*d_scen[id];
        }
}

__global__
void generate_rates_0_CIR(number * d_scen, unsigned nsim, number c, number l, number d, curandState * u, curandState * n)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                number chi=statistics_gpu::get_chi(d-1., &u[id], &n[id]);
                number random=(d_scen[id]+sqrt(l))*(d_scen[id]+sqrt(l))+chi;
                d_scen[id]=c*random;
        }
}

__global__
void generate_rates_i_CIR(number * d_scen, number * d_scen_old, unsigned nsim, number c, number e, number d,
                          curandState * u, curandState * n)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                number l=d_scen_old[id]*e/c;
                number chi=statistics_gpu::get_chi(d-1., &u[id], &n[id]);
                number random=(d_scen[id]+sqrt(l))*(d_scen[id]+sqrt(l))+chi;
                d_scen[id]=c*random;
        }
}

__global__
void evaluate_DF(number * d_DF, number * d_A, number * d_B, number * d_scen, unsigned pricing_grid_size,
                 unsigned nsim, unsigned i)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                for (unsigned j=0; j<pricing_grid_size; ++j)
                {
                        d_DF[id+j*nsim]=exp(-d_scen[id]*d_B[j])*d_A[j];
                }
        }
}

__global__
void evaluate_DF_V(number * d_DF, number * d_A, number * d_B, number * d_scen, unsigned pricing_grid_size,
                   unsigned nsim, unsigned i)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                for (unsigned j=0; j<pricing_grid_size; ++j)
                {
                        d_DF[id+j*nsim]=exp(-d_scen[id]*d_B[j]+d_A[j]);
                }
        }
}

__global__
void evaluate_MtM(number * d_MtM, number * d_DF, number * d_irsCF, unsigned i, unsigned nsim, unsigned pricing_grid_size, unsigned simulation_grid)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                for (unsigned j=i; j<simulation_grid+1; ++j)
                {
                        d_MtM[(i-1)*nsim+id]+=d_irsCF[j]*d_DF[id+(j-i)*nsim];
                }
        }
}

__global__
void evaluate_PFE(number * d_PFE, number * d_NPFE, number * d_MtM, unsigned nsim, number alpha)
{
        unsigned id=get_index();
        unsigned index_h=id*nsim+round(alpha*nsim);
        unsigned index_l=id*nsim+round((1.-alpha)*nsim);
        d_PFE[id]=max(d_MtM[index_h], static_cast<number>(0.));
        d_NPFE[id]=max(-d_MtM[index_l], static_cast<number>(0.));
}

void evaluate_cva(RateModels rate_model, MC_Type mc_type, unsigned nsim, unsigned simulation_grid, std::vector<number> const & A_, std::vector<number> const & B_, std::vector<number> & pricing_grid, std::vector<number> const & data,  std::vector<std::pair<number, number> > const & zero_coupon, std::vector<number> const & irs_CF, tools::Block_Threads & bt, unsigned num_gpus, number & EPE, number & NEPE, std::vector<number> & EE, std::vector<number> & NEE, std::vector<number> & PFE, std::vector<number> & NPFE, unsigned seed)
{
        using namespace thrust;
        using namespace std;

        std::pair<unsigned, unsigned> b_t(0,0);
        b_t=bt.evaluate_bt(nsim);

        vector<vector<number> > EE_(num_gpus, vector<number>(simulation_grid, 0.));
        vector<vector<number> > NEE_(num_gpus, vector<number>(simulation_grid, 0.));
        vector<vector<number> > PFE_(num_gpus, vector<number>(simulation_grid, 0.));
        vector<vector<number> > NPFE_(num_gpus, vector<number>(simulation_grid, 0.));

        omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
        #pragma omp parallel
        {
                unsigned cpu_thread_id = omp_get_thread_num();
                cudaSetDevice(cpu_thread_id);

                vector<number> my_pricing_grid(pricing_grid);

                // Generating random numbers
                device_vector<number> scen(simulation_grid*nsim);
                number * d_scen=raw_pointer_cast(&scen[0]);
                number spot_rate=zero_coupon[0].second;

                device_vector<number> A(A_), B(B_);
                number * d_A=raw_pointer_cast(&A[0]);
                number * d_B=raw_pointer_cast(&B[0]);

                device_vector<number> MtM(simulation_grid*nsim, -1.);
                number * d_MtM=raw_pointer_cast(&MtM[0]);

                device_vector<number> irs(irs_CF);
                number * d_irsCF=raw_pointer_cast(&irs[0]);

                curandState * devStates_u=NULL;
                curandState * devStates_n=NULL;

                if (rate_model==CIR)
                {
                        cudaMalloc(&devStates_u, nsim*sizeof(curandState));
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("cudaMalloc", __FILE__, __LINE__);
#endif
                        cudaMalloc(&devStates_n, nsim*sizeof(curandState));
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("cudaMalloc", __FILE__, __LINE__);
#endif

                        setup_curand<<<b_t.second, b_t.first>>>(devStates_u, seed, nsim);
#ifdef __CUDA_ERROR_CHECK__
                        cudaDeviceSynchronize();
                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                        setup_curand<<<b_t.second, b_t.first>>>(devStates_n, seed+1, nsim);
#ifdef __CUDA_ERROR_CHECK__
                        cudaDeviceSynchronize();
                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                }

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
                        curandGenerateNormal(gen, d_scen, simulation_grid*nsim, 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandGenerateNormal", __FILE__, __LINE__);
#endif

#else
                        curandGenerateNormalDouble(gen, d_scen, simulation_grid*nsim, 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandGenerateNormalDouble", __FILE__, __LINE__);
#endif

#endif
                        curandDestroyGenerator(gen);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandDestroyGenerator", __FILE__, __LINE__);
#endif
                }
                else
                {
                        curandGenerator_t gen;
                        // Create pseudo-random number generator
                        curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandCreateGenerator", __FILE__, __LINE__);
#endif
                        // Set seed
                        curandSetQuasiRandomGeneratorDimensions(gen, simulation_grid);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandCreateGenerator", __FILE__, __LINE__);
#endif
                        // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                        curandGenerateNormal(gen, d_scen, simulation_grid*nsim, 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandGenerateNormal", __FILE__, __LINE__);
#endif

#else
                        curandGenerateNormalDouble(gen, d_scen, simulation_grid*nsim, 0., 1.);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandGenerateNormalDouble", __FILE__, __LINE__);
#endif

#endif
                        curandDestroyGenerator(gen);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("curandDestroyGenerator", __FILE__, __LINE__);
#endif
                }

                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        if (rate_model==HullWhite)
                        {
                                if (i==0)
                                {
                                        generate_rates_0<<<b_t.second, b_t.first>>>(&d_scen[0], nsim, spot_rate, data[simulation_grid+i],
                                                        data[i], data[2*simulation_grid+i]);
#ifdef __CUDA_ERROR_CHECK__
                                        cudaDeviceSynchronize();
                                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                                }
                                else
                                {
                                        generate_rates_i<<<b_t.second, b_t.first>>>(&d_scen[i*nsim], &d_scen[(i-1)*nsim], nsim, spot_rate,
                                                        data[simulation_grid+i], data[i], data[2*simulation_grid+i]);
#ifdef __CUDA_ERROR_CHECK__
                                        cudaDeviceSynchronize();
                                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                                }
                        }
                        else if (rate_model==Vasicek)
                        {
                                if (i==0)
                                {
                                        generate_rates_0<<<b_t.second, b_t.first>>>(&d_scen[0], nsim, spot_rate, data[1],
                                                        data[0], data[2]);
#ifdef __CUDA_ERROR_CHECK__
                                        cudaDeviceSynchronize();
                                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                                }
                                else
                                {
                                        generate_rates_i<<<b_t.second, b_t.first>>>(&d_scen[i*nsim], &d_scen[(i-1)*nsim], nsim, spot_rate,
                                                        data[1], data[0], data[2]);
#ifdef __CUDA_ERROR_CHECK__
                                        cudaDeviceSynchronize();
                                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                                }
                        }
                        else if (rate_model==CIR)
                        {
                                if (i==0)
                                {
                                        number l=spot_rate*data[simulation_grid+i]/data[i];
                                        generate_rates_0_CIR<<<b_t.second, b_t.first>>>(&d_scen[0], nsim, data[i], l,
                                                        data[2*simulation_grid+1], devStates_u, devStates_n);
#ifdef __CUDA_ERROR_CHECK__
                                        cudaDeviceSynchronize();
                                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                                }
                                else
                                {
                                        generate_rates_i_CIR<<<b_t.second, b_t.first>>>(&d_scen[i*nsim], &d_scen[(i-1)*nsim], nsim, data[i],
                                                        data[simulation_grid+i], data[2*simulation_grid+1], devStates_u, devStates_n);
#ifdef __CUDA_ERROR_CHECK__
                                        cudaDeviceSynchronize();
                                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                                }
                        }
                        else
                        {
                                throw(std::logic_error("Model unknown."));
                        }

                        my_pricing_grid.pop_back();

                        device_vector<number> DF_vec (my_pricing_grid.size()*nsim);
                        number * d_DF=raw_pointer_cast(&DF_vec[0]);

                        if (rate_model==Vasicek)
                        {
                                evaluate_DF_V<<<b_t.second, b_t.first>>>(d_DF, &d_A[i*simulation_grid],
                                                &d_B[i*simulation_grid],
                                                &d_scen[i*nsim],
                                                my_pricing_grid.size(), nsim, i);
#ifdef __CUDA_ERROR_CHECK__
                                cudaDeviceSynchronize();
                                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                        }
                        else
                        {
                                evaluate_DF<<<b_t.second, b_t.first>>>(d_DF, &d_A[i*simulation_grid],
                                                                       &d_B[i*simulation_grid],
                                                                       &d_scen[i*nsim],
                                                                       my_pricing_grid.size(), nsim, i);
#ifdef __CUDA_ERROR_CHECK__
                                cudaDeviceSynchronize();
                                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif
                        }

                        evaluate_MtM<<<b_t.second, b_t.first>>>(d_MtM, d_DF, d_irsCF, i+1, nsim,
                                                                my_pricing_grid.size(), simulation_grid);
#ifdef __CUDA_ERROR_CHECK__
                        cudaDeviceSynchronize();
                        tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                }

                if (rate_model==CIR)
                {
                        cudaFree(devStates_u);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("cudaFree", __FILE__, __LINE__);
#endif
                        cudaFree(devStates_n);
#ifdef __CUDA_ERROR_CHECK__
                        tools_gpu::checkCUDAError("cudaFree", __FILE__, __LINE__);
#endif
                }

                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        thrust::sort(MtM.begin()+i*nsim, MtM.begin()+(i+1)*nsim);
                }

                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        EE_[cpu_thread_id][i]=thrust::transform_reduce(MtM.begin()+i*nsim,
                                              MtM.begin()+(i+1)*nsim,
                                              positive_value<number>(), 0.,
                                              thrust::plus<number>());
                        NEE_[cpu_thread_id][i]=-thrust::transform_reduce(MtM.begin()+i*nsim,
                                               MtM.begin()+(i+1)*nsim,
                                               negative_value<number>(), 0.,
                                               thrust::plus<number>());

                }

                vector<number> PFE2(simulation_grid);
                device_vector<number> PFE_vector(simulation_grid);
                number * d_PFE=raw_pointer_cast(&PFE_vector[0]);
                vector<number> NPFE2(simulation_grid);
                device_vector<number> NPFE_vector(simulation_grid);
                number * d_NPFE=raw_pointer_cast(&NPFE_vector[0]);

                evaluate_PFE<<<1, simulation_grid>>>(d_PFE, d_NPFE, d_MtM, nsim, Constants::quantile);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                cudaMemcpy(&PFE2[0], d_PFE, simulation_grid*sizeof(number),
                           cudaMemcpyDeviceToHost);
#ifdef __CUDA_ERROR_CHECK__
                tools_gpu::checkCUDAError("cudaMemcpy", __FILE__, __LINE__);
#endif
                cudaMemcpy(&NPFE2[0], d_NPFE, simulation_grid*sizeof(number),
                           cudaMemcpyDeviceToHost);
#ifdef __CUDA_ERROR_CHECK__
                tools_gpu::checkCUDAError("cudaMemcpy", __FILE__, __LINE__);
#endif

                PFE_[cpu_thread_id]=PFE2;
                NPFE_[cpu_thread_id]=NPFE2;

                cudaDeviceSynchronize();

        }

        for (unsigned i=0; i<simulation_grid; ++i)
        {
                for (unsigned j=0; j<num_gpus; ++j)
                {
                        EE[i]+=EE_[j][i]/(nsim*num_gpus);
                        NEE[i]+=NEE_[j][i]/(nsim*num_gpus);
                        PFE[i]+=PFE_[j][i]/num_gpus;
                        NPFE[i]+=NPFE_[j][i]/num_gpus;
                }
                EPE+=EE[i]/simulation_grid;
                NEPE+=NEE[i]/simulation_grid;
        }

}

}
