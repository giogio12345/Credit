#ifndef __cdo_hpp
#define __cdo_hpp

#include "CreditDerivative.hpp"
#include "Cdo_gpu.cuh"

//! This class evaluate the price of a Collateralized Debt Obbligation
/*! This class inherits from the CreditDerivative object and calculates the value of a Synthetic CDO and its Break-Even Spread. The algorithm is taken from http://math.nyu.edu/~atm262/spring06/ircm/cdo/ .
 */
template <CopulaType c_type>
class Cdo: public CreditDerivative<c_type>
{
protected:
        number          a;
        number          d;
        number          c;
        bool            isPB;

        virtual void            run_cpu();
        virtual void            run_gpu();
        virtual inline unsigned timetoperiod(number t, unsigned maxidx, number daycount)
        {
                return std::min(unsigned(floor(t/daycount)) + 1, maxidx);
        }
        virtual inline number price()
        {
                return 10000*this->results[3];
        }

public:
        //! Constructor
        /*! This is the constructor of the class Cdo. The argument list is the following:
         * \param type_         type of calculation: CPU or GPU
         * \param mc_type_      type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
         * \param dim_          dimension of the reference portfolio
         * \param rho_          correlation index in the reference portfolio. This index must be in the following set:\f[ \rho\in\left(\frac{-1}{dim-1},1\right).\f]
         * \param lambda_       hazard rate
         * \param R_            recovery rate
         * \param npf_          notional per firm
         * \param a_            attachment point
         * \param d_            detachment point
         * \param c_            coupon on fixed leg
         * \param r_            risk-free interest rate
         * \param T_            maturity
         * \param daycount_     coupon frequency per year (e.g. 0.5 for semi-annual, 0.25 for quarterly payments)
         * \param isPB          true for ProtectionBuyer, false for ProtectionSeller (just changes the sign of the contract value)
         * \param nsim_         number of simulation
         * \param max_gpus      maximum number of usable GPUs (default to 1)
         * \param dof           degrees of freedom of the Student's t copula (default to 8).
         */
        Cdo(Type type_,
            MC_Type mc_type_,
            unsigned dim_,
            number rho_,
            number lambda_,
            number R_,
            number npf_,
            number a_,
            number d_,
            number c_,
            number r_,
            number T_,
            number daycount_,
            bool isPB_,
            unsigned nsim_,
            unsigned max_gpus=1,
            unsigned dof=8);

        //!
        /*! This function returns the confidence interval of the price.
         */
        virtual inline std::pair<number, number>       get_ic() const
        {
                return std::make_pair(10000*(this->mc_type==MC?this->results[3]-
                                       Constants::alpha*this->results[7]:0.),
                                      10000*(this->mc_type==MC?this->results[3]+
                                       Constants::alpha*this->results[7]:0.));
        }

        //! Overload of the ostream operator for the class Cdo
        /*! This function prints all the evaluated data for the CDO, i.e. the values of the fixed and the floating leg (Vfix, Vflt), the total value of the contract (VCdo), the break-even spread (Cbe), the correlation between the fixed and the floating leg and the elapsed time.
         */
        template <CopulaType c_type_>
        friend std::ostream & operator<<(std::ostream & out, const Cdo<c_type_> & cdo);
};

template <CopulaType c_type>
Cdo<c_type>::Cdo(Type type_,
                 MC_Type mc_type_,
                 unsigned dim_,
                 number rho_,
                 number lambda_,
                 number R_,
                 number npf_,
                 number a_,
                 number d_,
                 number c_,
                 number r_,
                 number T_,
                 number daycount_,
                 bool isPB_,
                 unsigned nsim_,
                 unsigned max_gpus,
                 unsigned dof)
        :
        CreditDerivative<c_type>(type_, mc_type_, dim_, rho_, lambda_,
                                R_, npf_, r_, T_, daycount_, nsim_, max_gpus, dof),
        a(a_),
        d(d_),
        c(c_),
        isPB(isPB_)
{}

template <CopulaType c_type>
void Cdo<c_type>::run_cpu()
{
        using namespace std;
        using namespace Eigen;

        //cout<<*m<<"\n";

        unsigned aidx = (unsigned)floor((a * this->dim / (1 - this->R)) + 1);
        unsigned didx = (unsigned)floor((d * this->dim / (1 - this->R)) + 1);
        if (a == (1-this->R))
        {
                aidx = this->dim;
        }
        if (d >= (1-this->R))
        {
                didx = this->dim;
        }
        assert(aidx >= 1);
        assert(didx <= this->dim);

        unsigned maxidx = (unsigned)(this->T/this->daycount) + 1;
        tools::Matrix NDef(maxidx, this->nsim);
        NDef.setZero();
        number fraction;
        number dfrac = d * this->dim / (1 - this->R);
        number afrac = a * this->dim / (1 - this->R);

        for (unsigned idx = aidx-1; idx < didx; idx++)
        {
                fraction = min(dfrac, (number)(idx+1)) - max((number)(idx), afrac);
                for (unsigned path = 0; path < this->nsim; path++)
                {
                        NDef(timetoperiod((*(this->m))(idx, path), maxidx, this->daycount)-1, path)
                        += fraction;
                        (*(this->m))(idx, path) = fraction * this->npf * (1 - this->R) *
                                                  exp(-this->r * (*(this->m))(idx, path)) *
                                                  ((*(this->m))(idx, path) <= this->T ? 1 : 0);
                }
        }

        tools::Vector Vflt(this->nsim);
        Vflt.setZero();

        for (unsigned j=aidx-1; j<didx; ++j)
        {
                for (unsigned i=0; i<this->nsim; ++i)
                {
                        Vflt(i)+=(*(this->m))(j,i);
                }
        }

        tools::Vector Vfix(this->nsim);
        Vfix.setZero();
        tools::Vector LastNotional(this->nsim);
        LastNotional.setOnes();
        LastNotional=LastNotional*(d-a)*this->npf*this->dim;
        tools::Vector Notional(this->nsim);
        Notional = LastNotional;


        for (unsigned period = 1; period < maxidx; period++)
        {
                tools::Vector vec=NDef.row(period-1);
                Notional -= vec * this->npf * (1 - this->R);
                Vfix += c * this->daycount * ((Notional + LastNotional) / 2) * exp(-this->r * period * this->daycount);
                LastNotional = Notional;
        }

        // expected values
        number EVflt=0.; //= Vflt.Sum() / paths;
        number EVfix=0.; //= Vfix.Sum() / paths;

        for (unsigned i=0; i<this->nsim; ++i)
        {
                EVflt+=Vflt(i)/this->nsim;
                EVfix+=Vfix(i)/this->nsim;
        }

        // calculate Vcdo, Cbe using expected values of Vflt and Vfix
        number EVcdo = (isPB ? 1 : -1) * (EVflt - EVfix);
        number ECbe = EVflt / (EVfix / this->c);

        number VVflt=0., VVfix=0.;

        for (unsigned i=0; i<this->nsim; ++i)
        {
                VVflt+=Vflt(i)*Vflt(i)/this->nsim;
                VVfix+=Vfix(i)*Vfix(i)/this->nsim;
        }

        VVflt-=EVflt * EVflt;
        VVfix-=EVfix * EVfix;

        number CVfltVfix = (Vflt.dot(Vfix))/this->nsim - EVflt * EVfix;
        number VVcdo = VVflt + VVfix - 2 * CVfltVfix;
        number VCbe = c * c * (EVflt / EVfix) * (EVflt / EVfix) *
                      (VVflt / (EVflt * EVflt) + VVfix / (EVfix * EVfix) - (2 * CVfltVfix) / (EVflt * EVfix));

        // standard errors
        number SVflt = sqrt(VVflt / this->nsim);
        number SVfix = sqrt(VVfix / this->nsim);
        number SVcdo = sqrt(VVcdo / this->nsim);
        number SCbe = sqrt(VCbe / this->nsim);

        this->results[0]=EVflt;
        this->results[1]=EVfix;
        this->results[2]=EVcdo;
        this->results[3]=ECbe;
        this->results[4]=SVflt;
        this->results[5]=SVfix;
        this->results[6]=SVcdo;
        this->results[7]=SCbe;
        this->results.push_back(CVfltVfix/(sqrt(VVflt) * sqrt(VVfix)));

        return;

}

template <CopulaType c_type>
void Cdo<c_type>::run_gpu()
{
        cdo_gpu::evaluate_cdo(this->d_T, this->dim, this->R, this->npf, a, d, c,
                              this->r, this->T, this->daycount, isPB, this->nsim,
                              this->bt, this->results, this->number_of_gpu);
}

template <CopulaType c_type>
std::ostream & operator<<(std::ostream & out, const Cdo<c_type> & cdo)
{
        out<<"Results for object "<<cdo.id<<":\n";
        out << "Vflt = " << cdo.results[0] << " +/- " << Constants::alpha * cdo.results[4] << "\n";
        out << "Vfix = " << cdo.results[1] << " +/- " << Constants::alpha * cdo.results[5] << "\n";
        out << "Vcdo = " << cdo.results[2] << " +/- " << Constants::alpha * cdo.results[6] << "\n";
        out << "Cbe  = " << cdo.results[3] << " +/- " << Constants::alpha * cdo.results[7] << "\n";
        out << "corr(Vflt,Vfix) = " << cdo.results[8] << "\n";
        out << "Time elapsed = " << cdo.time <<"ms.\n";
        out << "******\n";
        return out;
}

#endif