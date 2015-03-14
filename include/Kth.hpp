#ifndef __kth_hpp
#define __kth_hpp

#include "CreditDerivative.hpp"
#include "Kth_gpu.cuh"

//! This class evaluate the price of a Kth-to-Default Swap
/*! This class inherits from the CreditDerivative object and calculates the spread of a Kth-to-Default Swap. The algorithm is taken from "Fathi Abid and Nader Naifar. Copula based simulation procedures for pricing basket credit derivatives. Munich Personal RePEc Archive, 2007" and "Stefano S. Galliani. Copula Functions and theis Application in Pricing and Risk Managing Multiname Credit Derivatives Products. Department of Mathematics, King’s College London, 2003".
 */
template <CopulaType c_type>
class Kth_to_Default: public CreditDerivative<c_type>
{
protected:
        unsigned                Kth;
        unsigned                ntime_substeps;

        virtual void            run_cpu();
        virtual void            run_gpu();
        virtual inline number   price()
        {
                return this->results[0];
        }
public:
        //! Constructor
        /*! This is the constructor of the class Cdo. The argument list is the following:
         * \param type_         type of calculation: CPU or GPU
         * \param mc_type_      type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
         * \param dim_          dimension of the reference portfolio
         * \param rho_          correlation index in the reference portfolio. This index must be in the following set:\f[ \rho\in\left(\frac{-1}{dim-1},1\right).\f]
         * \param lambda_       hazard rate
         * \param Kth_          seniority of the contract, e.g. 1 for a First-to-Default Swap, 2 for a Second-to-Default Swap, ...
         * \param R_            recovery rate
         * \param npf_          notional per firm
         * \param r_            risk-free interest rate
         * \param T_            maturity
         * \param daycount_     coupon frequency per year (e.g. 0.5 for semi-annual, 0.25 for quarterly payments)
         * \param nsim_         number of simulation
         * \param max_gpus      maximum number of usable GPUs (default to 1)
         * \param dof           degrees of freedom of the Student's t copula (default to 8).
         */
        Kth_to_Default(Type type_,
                       MC_Type mc_type_,
                       unsigned dim_,
                       number rho_,
                       number lambda_,
                       unsigned Kth_,
                       number R_,
                       number npf_,
                       number r_,
                       number T_,
                       number daycount_,
                       unsigned nsim_,
                       unsigned max_gpus=1,
                       unsigned dof=8);

        //! Overload of the ostream operator for the class Kth_to_Default
        /*! This function prints all the evaluated data for the Kth-to-Default Swap, i.e. the spread and the quarterly payment, the value of Premium Leg, Default Leg and Accrued Premium, and the elapsed time.
         */
        template <CopulaType c_type_>
        friend std::ostream & operator<<(std::ostream & out, const Kth_to_Default<c_type_> & kth);
};

template <CopulaType c_type>
Kth_to_Default<c_type>::Kth_to_Default(Type type_,
                                       MC_Type mc_type_,
                                       unsigned dim_,
                                       number rho_,
                                       number lambda_,
                                       unsigned Kth_,
                                       number R_,
                                       number npf_,
                                       number r_,
                                       number T_,
                                       number daycount_,
                                       unsigned nsim_,
                                       unsigned max_gpus,
                                       unsigned dof)
        :
        CreditDerivative<c_type>(type_, mc_type_, dim_, rho_, lambda_,
                                R_, npf_, r_, T_, daycount_, nsim_, max_gpus, dof),
        Kth(Kth_),
        ntime_substeps(10)
{}

template <CopulaType c_type>
void Kth_to_Default<c_type>::run_cpu()
{
        using namespace std;
        using namespace Eigen;

        unsigned n=static_cast<unsigned> (round(this->T/this->daycount));
        number dt=this->daycount;
        number dt2=dt/static_cast<number>(ntime_substeps);

        number t(dt);
        //number B(0.);
        //number SurvivalProb(0.);

        vector<vector<bool> >   SP(n, vector<bool>(this->nsim, false));
        vector<number>          DP_vec(this->nsim);
        vector<number>          AP_vec(this->nsim, 0.);

        for (unsigned i=0; i<n; ++i)
        {
                //B=exp(-r*t);
                vector<unsigned> n_default(this->nsim, 0);
                //unsigned temp=0;
                for (unsigned j=0; j<this->nsim; ++j)
                {
                        for (unsigned k=0; k<this->dim; ++k)
                        {
                                if ((*(this->m))(k,j)<=t)
                                {
                                        ++(n_default[j]);
                                }
                        }
                        if (n_default[j]>=Kth)
                        {
                                SP[i][j]=true;
                        }
                }
                t+=dt;
        }

        for (unsigned i=0; i<this->nsim; ++i)
        {
                if ((*(this->m))(Kth-1, i)<this->T)
                {
                        DP_vec[i]=(1-this->R)*exp(-this->r*(*(this->m))(Kth-1, i))/this->nsim;
                }
        }

        t=0.;
        for (unsigned i=0; i<n; ++i)
        {
                number current_time(t);
                for (unsigned j=0; j<ntime_substeps; ++j)
                {
                        // conto quanti sono defautati fra current_time e current_time+dt2
                        //number ProbOfDefault(0.);
                        for (unsigned k=0; k<this->nsim; ++k)
                        {
                                if ((*(this->m))(Kth-1, k)>current_time &&
                                                (*(this->m))(Kth-1, k)<=current_time+dt2)
                                {
                                        AP_vec[k]+=(current_time+dt2/2.-t)
                                                   *exp(-this->r*(current_time+dt2/2.))/this->nsim;
                                }
                        }
                        current_time+=dt2;
                }
                t+=dt;
        }

        vector<number> SP_E(n, 0.);
        vector<number> SP_V(n, 0.);

        for (unsigned i=0; i<n; ++i)
        {
                for (unsigned j=0; j<this->nsim; ++j)
                {
                        if (SP[i][j])
                        {
                                SP_E[i]+=1./this->nsim;
                        }
                }
                SP_E[i]=1-SP_E[i];
        }

        number PL_E=0.;
        number DP_E=0.;
        number AP_E=0.;

        // PL
        t=dt;
        for (unsigned i=0; i<n; ++i)
        {
                PL_E+=this->daycount*exp(-this->r*t)*SP_E[i];
                t+=dt;
        }

        // DP
        for (unsigned i=0; i<this->nsim; ++i)
        {
                DP_E+=DP_vec[i];
        }

        // AP
        for (unsigned i=0; i<this->nsim; ++i)
        {
                AP_E+=AP_vec[i];
        }

        number spread_E=DP_E/(PL_E+AP_E);

        this->results[0]=10000*spread_E;//this->npf*this->dim*this->daycount*spread_E;
        this->results[1]=0;
        this->results[2]=PL_E;
        this->results[3]=0;//PL_V;
        this->results[4]=DP_E;
        this->results[5]=0;//DP_V;
        this->results[6]=AP_E;
        this->results[7]=0;//AP_V;
}


template <CopulaType c_type>
void Kth_to_Default<c_type>::run_gpu()
{
        kth_gpu::evaluate_kth(Kth, this->d_T, this->dim, this->R, this->npf, this->r, this->T,
                              this->daycount, this->nsim, this->bt, this->results,
                              this->number_of_gpu);
}

template <CopulaType c_type>
std::ostream & operator<<(std::ostream & out, const Kth_to_Default<c_type> & kth)
{
        out<<"Results for object "<<kth.id<<":\n";
        out << "Spread = " << kth.results[0] << "\n";
        out << "Regular Premium = " << kth.results[0]*kth.dim * kth.npf * kth.daycount/10000. << "\n";
        out << "PL = " << kth.results[2] << "\n";
        out << "DP = " << kth.results[4] << "\n";
        out << "AP  = " << kth.results[6] << "\n";
        out << "Time elapsed = " << kth.time <<"ms.\n";
        out << "******\n";
        return out;
}

#endif