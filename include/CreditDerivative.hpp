#ifndef __Credit_Derivative_hpp
#define __Credit_Derivative_hpp

#include "DefTimesGen.hpp"

//! Abstract base class for CDO and Kth-to-Default classes
/*! This class stores the common elements of the classes Kth_to_Default and Cdo and defines their layout.
 */
template <CopulaType c_type>
class CreditDerivative
{
protected:
        Type                    type;
        MC_Type                 mc_type;

        unsigned                number_of_gpu;

        unsigned                dim;
        number                  rho;
        number                  lambda;

        number                  R;
        number                  npf;
        number                  r;
        number                  T;
        number                  daycount;

        unsigned                nsim;

        std::unique_ptr<DefTimesGen> def_times_obj;

        std::unique_ptr<tools::Matrix> m;
        std::vector<number *>   d_T;

        std::vector<number>     results;

        number                  time;

        tools::Block_Threads	bt;

        unsigned                id;

        virtual void            run_cpu()=0;
        virtual void            run_gpu()=0;
        virtual number          price()=0;

public:
        //! Constructor
        /*! Constructor called by derived classes.
         */
        CreditDerivative(Type type_,
                         MC_Type mc_type_,
                         unsigned dim_,
                         number rho_,
                         number lambda_,
                         number R_,
                         number npf_,
                         number r_,
                         number T_,
                         number daycount_,
                         unsigned nsim_,
                         unsigned max_gpus=1,
                         unsigned dof=8);

        //!
        /*! A function that allows the user to set the maximum number of threads per block.
         */
        virtual void set_max_threads_per_block(unsigned);

        //!
        /*! This function allows the user to print the matrix of default times in matlab format.
         */
        virtual void print_matrix();

        //!
        /*! This function allows the user to get the id of the object.
         */
        virtual inline unsigned get_id() const
        {
                return id;
        }

        //!
        /*! This function evaluate the price of the derivative.
         */
        virtual inline number run()
        {
                struct timeval real_s, real_e;
                gettimeofday(&real_s, NULL);
                if (type==CPU)
                {

                        run_cpu();
                }
                else
                {
                        run_gpu();
                }
                gettimeofday(&real_e, NULL);
                time+=(real_e.tv_sec-real_s.tv_sec)*1.e3+(real_e.tv_usec - real_s.tv_usec)/1.e3;
                return price();
        };

        //!
        /*! This function returns the time elapsed during the generation of the scenarios and the calculation of the price.
         */
        virtual inline number   get_time() const
        {
                return time;
        };

        //!
        /*! This function returns a vector storing the results of the calculation.
         */
        virtual inline std::vector<number> get_results() const
        {
                return results;
        };
};

template <CopulaType c_type>
CreditDerivative<c_type>::CreditDerivative(Type type_,
                MC_Type mc_type_,
                unsigned dim_,
                number rho_,
                number lambda_,
                number R_,
                number npf_,
                number r_,
                number T_,
                number daycount_,
                unsigned nsim_,
                unsigned max_gpus,
                unsigned dof)
        :
        type(type_),
        mc_type(mc_type_),
        number_of_gpu(std::min(max_gpus, tools_gpu::get_device_count())),
        dim(dim_),
        rho(rho_),
        lambda(lambda_),
        R(R_),
        npf(npf_),
        r(r_),
        T(T_),
        daycount(daycount_),
        nsim(nsim_),
        d_T(std::vector<number *> (number_of_gpu, NULL)),
        results(8, 0.),
        time(0.),
        bt()
{
        struct timeval real_s, real_e;

        tools::Counter::counter()++;
        id=tools::Counter::counter();

        if (id==1)
        {
                if (system(NULL))
                {
                        if(system("mkdir -p matlab"));
                }
                else
                {
                        std::exit(-1);
                }
        }

        if (type==CPU)
        {
                if (c_type==Gaussian)
                {
                        def_times_obj=std::unique_ptr<DefTimesGen>(new DefTimesGen_GaussianCopula(type, mc_type, dim, rho, lambda));
                }
                else
                {
                        def_times_obj=std::unique_ptr<DefTimesGen>(new DefTimesGen_tCopula(type, mc_type, dim, rho, lambda, dof));
                }
                gettimeofday(&real_s, NULL);
                m=def_times_obj->generate_deftimes_CPU(nsim);
        }
        else
        {
                nsim/=number_of_gpu;
                if (c_type==Gaussian)
                {
                        def_times_obj=std::unique_ptr<DefTimesGen>(new DefTimesGen_GaussianCopula(type, mc_type, dim, rho, lambda, number_of_gpu));
                }
                else
                {
                        def_times_obj=std::unique_ptr<DefTimesGen>(new DefTimesGen_tCopula(type, mc_type, dim, rho, lambda, number_of_gpu, dof));
                }
                gettimeofday(&real_s, NULL);
                d_T=def_times_obj->generate_deftimes_GPU(nsim);
        }

        gettimeofday(&real_e, NULL);
        time+=(real_e.tv_sec-real_s.tv_sec)*1.e3+(real_e.tv_usec - real_s.tv_usec)/1.e3;

}

template <CopulaType c_type>
void CreditDerivative<c_type>::set_max_threads_per_block(unsigned max_threads)
{
        if (type!=GPU)
        {
                throw(std::logic_error("This constructor must be used for GPU calculations."));
        }
        else
        {
                bt.set_max_thread_per_block(max_threads);
                def_times_obj->set_max_threads_per_block(max_threads);
        }
}

template <CopulaType c_type>
void CreditDerivative<c_type>::print_matrix()
{
        if (type==GPU)
        {
                throw(std::logic_error("This function is not working on GPUs."));
        }

        std::ofstream stream;
        std::string name("matlab/DefTimes");
        name.append(std::to_string(this->id));
        name.append(".m");
        stream.open(name);

        if (!stream.is_open())
        {
                throw(std::logic_error("Cannot open a file."));
        }
        stream<<"DefTimesMatrix=[ ";
        for (unsigned i=0; i<dim; ++i)
        {
                for (unsigned j=0; j<nsim; ++j)
                {
                        if (j!=nsim-1)
                        {
                                stream<<(*m)(i,j)<<", ";
                        }
                        else if (j==nsim-1 && i==dim-1)
                        {
                                stream<<(*m)(i,j)<<"];\n";
                        }
                        else
                        {
                                stream<<(*m)(i,j)<<"; ";
                        }
                }
        }
        stream.close();
}

#endif