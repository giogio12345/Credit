#include "DefTimesGen.hpp"

// class DefTimesGen

DefTimesGen::~DefTimesGen()
{
        if (type==GPU)
        {
                tools_gpu::cudaFree_wrapper(d_T, number_of_gpu);
        }
}

DefTimesGen::DefTimesGen(Type type_, MC_Type mc_type_, unsigned dim_, number rho_, number lambda_)
        :
        type(type_),
        mc_type(mc_type_),
        dim(dim_),
        rho(rho_),
        lambda(lambda_),
        d_T(std::vector<number *>(1,NULL)),
        number_of_gpu(0),
        engine(rd()),
        n_dist(0.,1.),
        bt()
{
        if (type!=CPU)
        {
                throw(std::logic_error("This constructor must be used for CPU calculations."));
        }
        if (rho<-1./(dim-1.) || rho>=1.)
        {
                throw(std::logic_error("The correlation matrix is not dp. rho must be in (-1/(dim-1),1)."));
        }
        S=tools::Matrix(dim, dim);
        compute_copula_matrix();
}

DefTimesGen::DefTimesGen(Type type_, MC_Type mc_type_, unsigned dim_, number rho_, number lambda_, unsigned num_gpus)
        :
        type(type_),
        mc_type(mc_type_),
        dim(dim_),
        rho(rho_),
        lambda(lambda_),
        number_of_gpu(num_gpus),
        engine(rd()),
        n_dist(0.,1.),
        bt()
{
        if (type!=GPU)
        {
                throw(std::logic_error("This constructor must be used for GPU calculations."));
        }
        if (rho<-1./(dim-1.) || rho>=1.)
        {
                throw(std::logic_error("The correlation matrix is not dp. rho must be in (-1/(dim-1),1)."));
        }
        if (num_gpus>static_cast<unsigned>(omp_get_num_procs()))
        {
                throw(std::logic_error("This is odd. Sounds like you have more GPUs than cores."));
        }
        S=tools::Matrix(dim, dim);
        compute_copula_matrix();
}

void DefTimesGen::set_max_threads_per_block(unsigned max_threads)
{
        if (type!=GPU)
        {
                throw(std::logic_error("This constructor must be used for GPU calculations."));
        }
        else
        {
                bt.set_max_thread_per_block(max_threads);
        }
}

void DefTimesGen::compute_copula_matrix()
{

        for (unsigned i=0; i<dim; ++i)
        {
                for (unsigned j=0; j<dim; ++j)
                {
                        if (i==j)
                        {
                                S(i,i)=1.;
                        }
                        else
                        {
                                S(i,j)=rho;
                        }
                }
        }
        Eigen::LLT<tools::Matrix> factorizator(S);
        S=factorizator.matrixL();

}

// class DefTimesGen_GaussianCopula

DefTimesGen_GaussianCopula::DefTimesGen_GaussianCopula(Type type_,
                MC_Type mc_type_,
                unsigned dim_,
                number rho_,
                number lambda_)
        :
        DefTimesGen(type_, mc_type_, dim_, rho_, lambda_)
{}

DefTimesGen_GaussianCopula::DefTimesGen_GaussianCopula(Type type_,
                MC_Type mc_type_,
                unsigned dim_,
                number rho_,
                number lambda_,
                unsigned num_gpus)
        :
        DefTimesGen(type_, mc_type_, dim_, rho_, lambda_, num_gpus)
{}

std::unique_ptr<tools::Matrix> DefTimesGen_GaussianCopula::generate_deftimes_CPU(unsigned nsim)
{

        ptr=std::unique_ptr<tools::Matrix> (new tools::Matrix(dim, nsim));

        if (mc_type==MC)
        {

                for (unsigned i=0; i<dim; ++i)
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                (*ptr)(i,j)=n_dist(engine);
                        }
                }

        }
        else
        {
                curandGenerator_t gen;
                // Create pseudo-random number generator
                curandCreateGeneratorHost(&gen, CURAND_RNG_QUASI_SOBOL64);
                // Set seed
                curandSetQuasiRandomGeneratorDimensions(gen, dim);
                // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                curandGenerateNormal(gen, (*ptr).data(), nsim*dim, 0., 1.);
#else
                curandGenerateNormalDouble(gen, (*ptr).data(), nsim*dim, 0., 1.);
#endif
                curandDestroyGenerator(gen);
        }

        *ptr=(S)*(*ptr);

        for (unsigned i=0; i<dim; ++i)
        {
                for (unsigned j=0; j<nsim; ++j)
                {
                        (*ptr)(i,j)=-log(statistics::phi((*ptr)(i,j)))/lambda;
                }
        }

        for (unsigned i=0; i<nsim; ++i)
        {
                auto col=(*ptr).col(i);
                auto begin = tools::index_begin<number>(col);
                auto end   = tools::index_end  <number>(col);
                std::sort(begin, end);
        }

        return std::move(ptr);


}

std::vector<number *> DefTimesGen_GaussianCopula::generate_deftimes_GPU(unsigned nsim)
{
        d_T=deftimes_gpu::generate_deftimes_guassian_GPU(mc_type, dim, nsim, lambda, S.data(), bt, number_of_gpu, rd());
        return d_T;
}


// class DefTimesGen_tCopula

DefTimesGen_tCopula::DefTimesGen_tCopula(Type type_,
                MC_Type mc_type_,
                unsigned dim_,
                number rho_,
                number lambda_,
                unsigned dof_)
        :
        DefTimesGen(type_, mc_type_, dim_, rho_, lambda_),
        dof(dof_),
        engine2(rd()),
        chi_dist(dof)
{}

DefTimesGen_tCopula::DefTimesGen_tCopula(Type type_,
                MC_Type mc_type_,
                unsigned dim_,
                number rho_,
                number lambda_,
                unsigned num_gpus,
                unsigned dof_)
        :
        DefTimesGen(type_, mc_type_, dim_, rho_, lambda_, num_gpus),
        dof(dof_),
        engine2(rd()),
        chi_dist(dof)
{}

std::unique_ptr<tools::Matrix> DefTimesGen_tCopula::generate_deftimes_CPU(unsigned nsim)
{
        using namespace statistics;

        ptr=std::unique_ptr<tools::Matrix> (new tools::Matrix(dim, nsim));
        std::vector<number> chi(nsim, 0.);

        number temp;

        if (mc_type==MC)
        {

                for (unsigned i=0; i<dim; ++i)
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                (*ptr)(i,j)=n_dist(engine);
                        }
                }
                for (unsigned i=0; i<nsim; ++i)
                {
                        chi[i]=chi_dist(engine2);
                }

        }

        else
        {
                std::vector<number> data(nsim*(dim+dof));

                curandGenerator_t gen;
                // Create pseudo-random number generator
                curandCreateGeneratorHost(&gen, CURAND_RNG_QUASI_SOBOL64);
                // Set seed
                curandSetQuasiRandomGeneratorDimensions(gen, dim+dof);
                // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                curandGenerateNormal(gen, data.data(), nsim*(dim+dof), 0., 1.);
#else
                curandGenerateNormalDouble(gen, data.data(), nsim*(dim+dof), 0., 1.);
#endif
                curandDestroyGenerator(gen);

                for (unsigned i=0; i<dim; ++i)
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                (*ptr)(i,j)=data[i*nsim+j];
                        }
                }
                for (unsigned i=0; i<nsim; ++i)
                {
                        for (unsigned j=0; j<dof; ++j)
                        {
                                temp=data[(dim+j)*nsim+i];
                                chi[i]+=temp*temp;
                        }
                        if (chi[i]<Constants::epsilon)
                        {
                                chi[i]+=Constants::epsilon;
                        }
                }
        }

        *ptr=(S)*(*ptr);

        int * dummy=new int[1];

        for (unsigned j=0; j<nsim; ++j)
        {
                for (unsigned i=0; i<dim; ++i)
                {
                        temp=(*ptr)(i,j)*sqrt(dof/chi[j]);
                        temp=tnc(temp, dof, 0., dummy);
                        (*ptr)(i,j)=-log(temp)/lambda;
                }
        }

        for (unsigned i=0; i<nsim; ++i)
        {
                auto col=(*ptr).col(i);
                auto begin = tools::index_begin<number>(col);
                auto end   = tools::index_end  <number>(col);
                std::sort(begin, end);
        }

        delete [] dummy;

        return std::move(ptr);


}

std::vector<number *> DefTimesGen_tCopula::generate_deftimes_GPU(unsigned nsim)
{
        d_T=deftimes_gpu::generate_deftimes_t_GPU(mc_type, dim, nsim, lambda, S.data(), dof, bt, number_of_gpu, rd());
        return d_T;
}
