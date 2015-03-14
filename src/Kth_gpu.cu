#include "Kth_gpu.cuh"

#define ntime_substeps 10

namespace kth_gpu
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

__global__ void evaluate_PL(unsigned * d_SP, number * d_T, unsigned n, unsigned Kth, number delta, unsigned dim, unsigned paths)
{
        unsigned id=get_index();
        if (id<paths)
        {
                number t=delta;
                for (unsigned j=0; j<n; ++j)
                {
                        unsigned defaulted=0;
                        // conto quanti sono defaultati fra 0 e t
                        for (unsigned k=0; k<dim; ++k)
                        {
                                if (d_T[id+k*paths]<=t)
                                {
                                        ++defaulted;
                                }
                        }
                        if (defaulted>=Kth)
                        {
                                d_SP[id+paths*j]=1;
                        }
                        t+=delta;
                }
        }
}

__global__ void evaluate_DP (number * d_T, number * d_DP, number r, number R, number T, unsigned Kth, unsigned dim, unsigned paths)
{
        unsigned id=get_index();
        if (id<paths)
        {
                if (d_T[id+(Kth-1)*paths]<T)
                {
                        d_DP[id]=(1-R)*exp(-r*d_T[id+(Kth-1)*paths])/paths;
                }
        }
}

__global__ void evaluate_AP (number * d_AP, number * d_T, unsigned Kth, unsigned n, unsigned paths, unsigned dim, number delta, number r)
{
        unsigned id=get_index();
        if (id<paths)
        {
                number DefProb[ntime_substeps];
                number t=0.;
                number dt2=delta/ntime_substeps;

                for (unsigned k=0; k<n; ++k)
                {
                        for (unsigned m=0; m<ntime_substeps; ++m)
                        {
                                DefProb[m]=0.f;
                        }
                        number current_time=t;
                        for (unsigned j=0; j<ntime_substeps; ++j)
                        {
                                if (d_T[id+(Kth-1)*paths]>current_time && d_T[id+(Kth-1)*paths]<=current_time+dt2)
                                {
                                        DefProb[j]+=1./paths;
                                }
                                current_time+=dt2/2.;
                                d_AP[id]+=(current_time-t)*exp(-r*current_time)*DefProb[j];
                                current_time+=dt2/2.;
                        }
                        t+=delta;
                }
        }
}

void evaluate_kth(unsigned Kth, std::vector<number *> d_T2, unsigned dim, number R, number npf, number r, number T, number daycount, unsigned paths, tools::Block_Threads & bt, std::vector<number> & results, unsigned num_gpus)
{
        using namespace std;
        using namespace thrust;

        unsigned n=static_cast<unsigned> (round(T/daycount));

        vector<vector<number> > SP(num_gpus, vector<number>(n, 1.));
        vector<number>		DP(num_gpus, 0.);
        vector<number>		AP(num_gpus, 0.);
        /*
        for (int i = 0; i < num_gpus; i++)
        {
                cudaDeviceProp dprop;
                cudaGetDeviceProperties(&dprop, i);
                printf("   %d: %s\n", i, dprop.name);
        }*/

        omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
        #pragma omp parallel
        {
                std::pair<unsigned, unsigned> b_t(0,0);
                b_t=bt.evaluate_bt(paths);

                unsigned cpu_thread_id = omp_get_thread_num();
                cudaSetDevice(cpu_thread_id);

                number * d_T=d_T2[cpu_thread_id];

                //cout<<"paths "<<paths<<"\nnumgpus "<<num_gpus<<"\nthread id "
                //    <<cpu_thread_id<<"\nnum_threads "<<omp_get_num_threads()<<"\n";

                device_vector<unsigned> SurvivalProb(paths*n, 0);
                unsigned * d_SP=raw_pointer_cast(&SurvivalProb[0]);

                // PL
                evaluate_PL<<<b_t.second, b_t.first>>> (d_SP, d_T, n, Kth, daycount, dim, paths);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                for (unsigned i=0; i<n; ++i)
                {
                        SP[cpu_thread_id][i]-=static_cast<number>(reduce(SurvivalProb.begin()+paths*i,
                                              SurvivalProb.begin()+paths*(i+1)))/paths;
                }

                // DP
                device_vector<number> DP_vector(paths, 0.);

                number * d_DP=raw_pointer_cast(&DP_vector[0]);

                evaluate_DP<<<b_t.second, b_t.first>>> (d_T, d_DP, r, R, T, Kth, dim, paths);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                DP[cpu_thread_id]=reduce(DP_vector.begin(), DP_vector.end(), 0.f);

                // AP
                device_vector<number> AP_vector(paths);

                number * d_AP=raw_pointer_cast(&AP_vector[0]);

                evaluate_AP<<<b_t.second, b_t.first>>> (d_AP, d_T, Kth, n, paths, dim, daycount, r);
#ifdef __CUDA_ERROR_CHECK__
                cudaDeviceSynchronize();
                tools_gpu::checkCUDAError("kernel execution", __FILE__, __LINE__);
#endif

                AP[cpu_thread_id]=reduce(AP_vector.begin(), AP_vector.end(), 0.f);

                cudaDeviceSynchronize();

        }

        number EPL=0.;
        number t=daycount;
        for (unsigned i=0; i<n; ++i)
        {
                for (unsigned j=0; j<num_gpus; ++j)
                {
                        EPL+=daycount*exp(-r*t)*SP[j][i]/num_gpus;
                }
                t+=daycount;
        }

        number EDP=0.;
        number EAP=0.;
        for (unsigned i=0; i<num_gpus; ++i)
        {
                EDP+=DP[i]/num_gpus;
                EAP+=AP[i]/num_gpus;
        }

        number spread_E=EDP/(EPL+EAP);

        results[0]=10000*spread_E;//npf*dim*daycount*spread_E;
        results[1]=0;
        results[2]=EPL;
        results[3]=0;
        results[4]=EDP;
        results[5]=0;
        results[6]=EAP;
        results[7]=0;

}


}
