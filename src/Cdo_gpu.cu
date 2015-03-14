#include "Cdo_gpu.cuh"

namespace cdo_gpu
{

__device__
unsigned timetoperiod(number t, unsigned maxidx, number daycount)
{
        return fminf(unsigned(floorf(t/daycount)) + 1, maxidx);
}

__device__
unsigned get_index()
{
        unsigned index_x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned index_y = blockIdx.y * blockDim.y + threadIdx.y;

        unsigned grid_width = gridDim.x * blockDim.x;
        unsigned index = index_y * grid_width + index_x;

        return index;
}

__global__ void evaluate_NDef    (	number * d_T, number * d_NDef, unsigned aidx, unsigned didx, unsigned maxidx,
                                        number daycount, number dfrac, number afrac, number Tmax, number R, number r,
                                        number npf, unsigned nsim, unsigned N)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                number fraction;
                for (unsigned idx = aidx-1; idx < didx; idx++)
                {
                        fraction=fmin(dfrac, static_cast<number>(idx+1))-
                                 fmax(static_cast<number>(idx), afrac);
                        d_NDef[id+nsim*(timetoperiod(d_T[id+idx*nsim], maxidx, daycount)-1)]+=fraction;
                        d_T[id+idx*nsim]=fraction * npf * (1 - R) *
                                         exp(-r * d_T[id+idx*nsim]) * (d_T[id+idx*nsim] <= Tmax ? 1 : 0);
                }
        }
}

__global__ void evaluate_Vflt (number * d_Vflt, number * d_VVflt, number * d_T, number aidx,
                               number didx, unsigned nsim)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                for (unsigned j=aidx-1; j<didx; ++j)
                {
                        d_Vflt[id]+=d_T[id+j*nsim];
                }
                d_VVflt[id]=d_Vflt[id]*d_Vflt[id]/nsim;
                d_Vflt[id]=d_Vflt[id]/nsim;
        }
}

__global__ void evaluate_Vfix(	number * d_Vfix, number * d_VVfix, number * d_T, number * d_NDef, number * d_Notional,
                                number * d_LastNotional, unsigned maxidx, number npf, number R, number r,
                                number daycount, number c, unsigned nsim)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                for (unsigned period = 1; period < maxidx; period++)
                {
                        d_Notional[id]-=d_NDef[id+(period-1)*nsim]*npf*(1-R);
                        d_Vfix[id] += c*daycount*((d_Notional[id]+d_LastNotional[id])/2.)*exp(-r*period*daycount);
                        d_LastNotional[id] = d_Notional[id];
                }
                d_VVfix[id]=d_Vfix[id]*d_Vfix[id]/nsim;
                d_Vfix[id]=d_Vfix[id]/nsim;
        }
}

__global__ void evaluate_CVfltVfix(number * d_CVfltVfix, number * d_Vflt, number * d_Vfix, unsigned nsim)
{
        unsigned id=get_index();
        if (id<nsim)
        {
                d_CVfltVfix[id]=nsim*d_Vflt[id]*d_Vfix[id];
        }
}

void evaluate_cdo(std::vector<number *> d_T2, unsigned N, number R, number npf, number a, number d, number c, number r, number Tmax, number daycount, bool isProtectionBuyer, unsigned nsim, tools::Block_Threads & bt, std::vector<number> & results, unsigned num_gpus)
{
        using namespace std;
        using namespace thrust;

        std::vector<number> 	Vflt(num_gpus, 0.), Vfix(num_gpus, 0.),
               VVflt(num_gpus, 0.), VVfix(num_gpus, 0.),
               CVfltVfix(num_gpus, 0.);

        std::pair<unsigned, unsigned> b_t(0,0);

        // calculate attachment and detachment points in terms of number of firms
        //cout << "calculate payments " << time(NULL) - lasttime << endl;
        unsigned aidx = (unsigned)floor((a * N / (1 - R)) + 1);
        unsigned didx = (unsigned)floor((d * N / (1 - R)) + 1);
        if (a == (1-R))
        {
                aidx = N;
        }
        if (d == (1-R))
        {
                didx = N;
        }
        assert(aidx >= 1);
        assert(didx <= N);

        // data structure for holding number of defaults in each coupon period
        // rows - coupon period, columns - sample nsim
        // (the last row is a catchall for defaults after Tmax)

        unsigned maxidx = (unsigned)(Tmax/daycount) + 1;
        number dfrac = d * N / (1 - R);
        number afrac = a * N / (1 - R);
        /*
        // display CPU and GPU configuration
        printf("number of host CPUs:\t%d\n", omp_get_num_procs());
        printf("number of CUDA devices:\t%d\n", num_gpus);

        for (int i = 0; i < num_gpus; i++)
        {
                cudaDeviceProp dprop;
                cudaGetDeviceProperties(&dprop, i);
                printf("   %d: %s\n", i, dprop.name);
        }*/

        omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
        #pragma omp parallel
        {
                unsigned cpu_thread_id = omp_get_thread_num();
                cudaSetDevice(cpu_thread_id);

                b_t=bt.evaluate_bt(nsim);

                number * d_T=d_T2[cpu_thread_id];

                device_vector<number> NDef(maxidx*nsim, 0.);
                number * d_NDef = raw_pointer_cast(&NDef[0]);

                evaluate_NDef <<<b_t.second, b_t.first>>> (	d_T, d_NDef, aidx, didx, maxidx, daycount,
                                dfrac, afrac, Tmax, R, r, npf, nsim, N);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                // calcolo Vflt
                device_vector<number> Vflt_vector(nsim, 0.);
                number * d_Vflt = raw_pointer_cast(&Vflt_vector[0]);
                device_vector<number> VVflt_vector(nsim, 0.);
                number * d_VVflt = raw_pointer_cast(&VVflt_vector[0]);

                evaluate_Vflt<<<b_t.second, b_t.first>>>(d_Vflt, d_VVflt, d_T, aidx, didx, nsim);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                Vflt[cpu_thread_id]=reduce(Vflt_vector.begin(), Vflt_vector.end(), 0.f);
                VVflt[cpu_thread_id]=reduce(VVflt_vector.begin(), VVflt_vector.end(), 0.f);

                // calcolo Vfix
                device_vector<number> Notional(nsim, (d-a)*npf*N);
                device_vector<number> LastNotional(nsim, (d-a)*npf*N);
                device_vector<number> Vfix_vector(nsim, 0.);
                device_vector<number> VVfix_vector(nsim, 0.);

                number * d_Vfix = raw_pointer_cast(&Vfix_vector[0]);
                number * d_VVfix = raw_pointer_cast(&VVfix_vector[0]);
                number * d_Notional = raw_pointer_cast(&Notional[0]);
                number * d_LastNotional = raw_pointer_cast(&LastNotional[0]);

                evaluate_Vfix<<<b_t.second, b_t.first>>>(	d_Vfix, d_VVfix, d_T, d_NDef, d_Notional, d_LastNotional,
                                maxidx, npf, R, r, daycount, c, nsim);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                Vfix[cpu_thread_id]=reduce(Vfix_vector.begin(), Vfix_vector.end(), 0.f);
                VVfix[cpu_thread_id]=reduce(VVfix_vector.begin(), VVfix_vector.end(), 0.f);

                // Calcolo CVfltfix
                device_vector<number> CVfltVfix_vector(nsim, 0.);
                number * d_CVfltVfix = raw_pointer_cast(&CVfltVfix_vector[0]);

                evaluate_CVfltVfix<<<b_t.second, b_t.first>>>(d_CVfltVfix, d_Vflt, d_Vfix, nsim);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                CVfltVfix[cpu_thread_id]=reduce(CVfltVfix_vector.begin(), CVfltVfix_vector.end(), 0.f);

                cudaDeviceSynchronize();
        }

        for (unsigned i=1; i<num_gpus; ++i)
        {
                Vflt[0]+=Vflt[i];
                VVflt[0]+=VVflt[i];

                Vfix[0]+=Vfix[i];
                VVfix[0]+=VVfix[i];

                CVfltVfix[0]+=CVfltVfix[i];
        }

        Vflt[0]/=num_gpus;
        Vfix[0]/=num_gpus;
        VVflt[0]/=num_gpus;
        VVfix[0]/=num_gpus;
        CVfltVfix[0]/=num_gpus;

        //cout<<Vfix[0]<<" "<<Vflt[0]<<" "<<CVfltVfix[0]<<"\n";

        //cout<<"nsim "<<nsim<<"\n";

        VVflt[0]-=Vflt[0]*Vflt[0];
        VVfix[0]-=Vfix[0]*Vfix[0];
        CVfltVfix[0]-=Vflt[0]*Vfix[0];

        number VVcdo=VVflt[0]+VVfix[0]-2*CVfltVfix[0];
        number VCbe = c * c * (Vflt[0] / Vfix[0]) * (Vflt[0] / Vfix[0]) *
                      (VVflt[0] / (Vflt[0] * Vflt[0]) + VVfix[0] / (Vfix[0] * Vfix[0]) -
                       (2 * CVfltVfix[0]) / (Vflt[0] * Vfix[0]));

        number EVcdo = (isProtectionBuyer ? 1 : -1) * (Vflt[0] - Vfix[0]);
        number ECbe = Vflt[0] / (Vfix[0] / c);

        // standard errors
        nsim*=num_gpus;
        number SVflt = sqrt(VVflt[0] / nsim);
        number SVfix = sqrt(VVfix[0] / nsim);
        number SVcdo = sqrt(VVcdo / nsim);
        number SCbe = sqrt(VCbe / nsim);

        results[0]=Vflt[0];
        results[1]=Vfix[0];
        results[2]=EVcdo;
        results[3]=ECbe;
        results[4]=SVflt;
        results[5]=SVfix;
        results[6]=SVcdo;
        results[7]=SCbe;
        results.push_back(CVfltVfix[0]/(sqrt(VVflt[0]) * sqrt(VVfix[0])));

}

}
