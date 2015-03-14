#ifndef __Cva_hpp
#define __Cva_hpp

#include <curand.h>
#include <iostream>
#include <fstream>
#include <string>

#include "RateModels.hpp"
#include "Cva_gpu.cuh"

// CVA di un IRC che riceve fisso e paga mobile
// con i tassi negativi sovrastimo la mia EE

//! This class evaluates the Expected Exposure and the Credit Valuation Adjustment of an Interest Rate Swap
/*! This class simulates the Mark to Market of a given Interest Rate Swap evaluating the Positive and Negative Expected Exposures (EE), Potential Future Exposures (PFE) and Expected Positive Exposures (EPE). Besides, given CDS term structures for one or both counterparties, it is able the evaluate the Bilateral Credit Valuation Adjustment of the swap. This class uses three models: Vasicek, Hull-White and Cox-Ingersol-Ros.
 */
template<RateModels rate_model>
class Cva
{
protected:
        Type                    type;           // CPU vs GPU
        MC_Type                 mc_type;        // MC vs QMC
        number                  T;
        number                  coupon;
        number                  a;
        number                  b;
        number                  sigma;
        number                  myR;
        number                  cR;

        unsigned                nsim;
        bool                    receiving_fix;
        unsigned                number_of_gpu;

        unsigned                simulation_grid;
        std::vector<number>     dt;
        std::vector<number>     pricing_grid;
        std::vector<number>     irs_CF;

        std::vector<number>     EE;
        std::vector<number>     NEE;
        std::vector<number>     PFE;
        std::vector<number>     NPFE;
        number                  EPE;
        number                  NEPE;
        number                  CVA;

        number                  spot_rate;

        std::vector<std::pair<number, number> > zero_coupon; // ZeroCoupon and dates

        number                  time;

        std::random_device		rd;
        std::mt19937			engine;
        std::normal_distribution<>	n_dist;

        std::vector<std::pair<number, number> > myCDS;
        std::vector<std::pair<number, number> > cCDS;

        std::vector<number> myCDS_interpolated;
        std::vector<number> cCDS_interpolated;
        std::vector<number> myDP;
        std::vector<number> cDP;

        tools::Block_Threads	bt;

        bool                    print_file;
        unsigned                id;

        void read_zcb(std::string const & zc_file);
        void read_CDS(std::string const & cds_file, bool mine);
        number alpha(number t);
        number linear_interpolation(std::vector<std::pair<number, number> > const & x, std::vector<number> const & y, number z);
        number interpolate(std::vector<std::pair<number, number> > const & x, number y);
        number zc_bondprices(number t);
        number evaluate_A(number t, number T_);
        number evaluate_B(number t, number T_);
        void ptfEval(std::vector<std::vector<number> > & MtM, number t, std::vector<std::vector<number> > const & DF, unsigned nsim, number c);
        void evaluateMtM(std::vector<std::vector<number> > & scen,
                         std::vector<std::vector<number> > & MtM,
                         std::ofstream & stream);
        void evaluate_cva();

        void run_cpu();
        void run_gpu();

public:
        //! Constructor for Hull-White
        /*! This is the constructor of the class Cva for the Hull-White model, given by: \f[ dr_{t}=\left(\theta(t)-ar_{t}\right)dt+\sigma dW_{t}.\f] \f$ a\f$ and \f$ \sigma \f$ are given by the user, \f$ \theta \f$ and \f$ r_0 \f$ are calculated with the zero coupon file. The argument list is the following:
         * \param type_                 type of calculation: CPU or GPU
         * \param mc_type_              type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
         * \param T_                    maturity of the swap
         * \param coupon_               fixed coupon of the swap
         * \param a_                    parameter \f$ a\f$ of the model
         * \param sigma_                parameter \f$ \sigma \f$ of the model
         * \param myR_                  recovery rate of the first counterparty
         * \param cR_                   recovery rate of the secondo counterparty
         * \param zc_file               zero coupon file (time convenction: act/365)
         * \param myCDS_file            CDS term structure of the first counterparty
         * \param cCDS_file             CDS term structure of the second counterparty
         * \param nsim_                 number of simulation
         * \param receiving_fix_        true for a receiving-fix swap, false for a paying-fix swap
         * \param max_gpus              maximum number of usable GPUs (default to 1)
         */
        Cva(Type type_,
            MC_Type mc_type_,
            number T_,
            number coupon_,
            number a_,
            number sigma_,
            number myR_,
            number cR_,
            std::string const & zc_file,
            std::string const & myCDS_file,
            std::string const & cCDS_file,
            unsigned nsim_,
            bool receiving_fix_,
            unsigned max_gpus=1);

        //! Constructor for Vasicek and CIR
        /*! This is the constructor of the class Cva for the Vasicek and the CIR model. The SDEs of these models are: \f[ dr_{t}=a(b-r_{t})dt+\sigma dW_{t},\f] for Vasicek and: \f[ dr_{t}=a(b-r_{t})dt+\sigma \sqrt{r_{t}}dW_{t},\f] for the CIR. \f$ a\f$, \f$ b\f$ and \f$ \sigma \f$ are given by the user, \f$ r_0 \f$ is calculated with the zero coupon file. The argument list is the following:
         * \param type_                 type of calculation: CPU or GPU
         * \param mc_type_              type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
         * \param T_                    maturity of the swap
         * \param coupon_               fixed coupon of the swap
         * \param a_                    parameter \f$ a\f$ of the model
         * \param b_                    parameter \f$ b\f$ of the model
         * \param sigma_                parameter \f$ \sigma \f$ of the model
         * \param myR_                  recovery rate of the first counterparty
         * \param cR_                   recovery rate of the secondo counterparty
         * \param zc_file               zero coupon file (time convenction: act/365)
         * \param myCDS_file            CDS term structure of the first counterparty
         * \param cCDS_file             CDS term structure of the second counterparty
         * \param nsim_                 number of simulation
         * \param receiving_fix_        true for a receiving-fix swap, false for a paying-fix swap
         * \param max_gpus              maximum number of usable GPUs (default to 1)
         * \note Quasi Monte Carlo method is not enabled for the CIR simulation.
         */
        Cva(Type type_,
            MC_Type mc_type_,
            number T_,
            number coupon_,
            number a_,
            number b_,
            number sigma_,
            number myR_,
            number cR_,
            std::string const & zc_file,
            std::string const & myCDS_file,
            std::string const & cCDS_file,
            unsigned nsim_,
            bool receiving_fix_,
            unsigned max_gpus=1);

        //!
        /*! A simple function the instructs the code the save in a Matlab file the interest rate and the Mark-to-Market simulations and all the results.
         */
        void set_printfile (bool print_file_)
        {
                if (type==GPU)
                {
                        std::cerr<<"Warning: this flag is not working on GPUs.\n";
                }
                print_file=print_file_;
        }

        //!
        /*! This function returns the Potential Future Exposure of the first counterparty
         */
        std::vector<number>     get_PFE()       const
        {
                return PFE;
        }

        //!
        /*! This function returns the Potential Future Exposure of the second counterparty, i.e. the Negative Potential Future Exposure of the first counterparty.
         */
        std::vector<number>     get_NPFE()       const
        {
                return NPFE;
        }

        //!
        /*! This function returns the Expected Exposure of the first counterparty
         */
        std::vector<number>     get_EE()       const
        {
                return EE;
        }

        //!
        /*! This function returns the Expected Exposure of the second counterparty, i.e. the Negative Expected Exposure of the first counterparty.
         */
        std::vector<number>     get_NEE()       const
        {
                return NEE;
        }

        //!
        /*! This function returns the Expected Positive Exposure of the first counterparty
         */
        number                  get_EPE()        const
        {
                return EPE;
        }

        //!
        /*! This function returns the Expected Positive Exposure of the second counterparty, i.e. the Negative Expected Positive Exposure of the first counterparty.
         */
        number                  get_NEPE()        const
        {
                return NEPE;
        }

        //!
        /*! This function returns the (Bilateral) Credit Valuation Ajustment of the swap.
         */
        number                  get_CVA()        const
        {
                return CVA;
        }

        //!
        /*! This function returns the elapsed time.
         */
        inline number           get_time()      const
        {
                return time;
        }

        //!
        /*! This function returns the id of this object
         */
        inline unsigned         get_id()        const
        {
                return id;
        }

        //!
        /*! This function runs the simulation and evaluates the MtM and the Expected Exposure.
         */
        void run();


        //! Overload of the ostream operator
        /*! This function prints all the evaluated data for the Expected Exposure and CVA, i.e. Positive and Negative Expected Exposures (EE), Potential Future Exposures (PFE), Expected Positive Exposures (EPE), the CVA and the elapsed time.
         */
        template <RateModels rate_model_>
        friend std::ostream & operator<<(std::ostream &, const Cva<rate_model_> &);
};

template<>
Cva<HullWhite>::Cva(Type type_,
                    MC_Type mc_type_,
                    number T_,
                    number coupon_,
                    number a_,
                    number sigma_,
                    number myR_,
                    number cR_,
                    std::string const & zc_file,
                    std::string const & myCDS_file,
                    std::string const & cCDS_file,
                    unsigned nsim_,
                    bool receiving_fix_,
                    unsigned max_gpus)
        :
        type(type_),
        mc_type(mc_type_),
        T(T_),
        coupon(coupon_),
        a(a_),
        b(0),
        sigma(sigma_),
        myR(myR_),
        cR(cR_),
        nsim(nsim_),
        receiving_fix(receiving_fix_),
        number_of_gpu(std::min(max_gpus, tools_gpu::get_device_count())),
        simulation_grid(T-1),
        dt(simulation_grid+1, 1.),
        pricing_grid(simulation_grid+1),
        irs_CF(simulation_grid+1, coupon),
        EE(simulation_grid, 0.),
        NEE(simulation_grid, 0.),
        PFE(simulation_grid, 0.),
        NPFE(simulation_grid, 0.),
        EPE(0.),
        NEPE(0.),
        CVA(0.),
        time(0.),
        engine(rd()),
        n_dist(0.,1.),
        myCDS_interpolated(simulation_grid, 0.),
        cCDS_interpolated(simulation_grid, 0.),
        myDP(simulation_grid, 0.),
        cDP(simulation_grid, 0.),
        bt(),
        print_file(false)
{
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

        read_zcb(zc_file);
        irs_CF[irs_CF.size()-1]+=1.;
        read_CDS(myCDS_file, true);
        read_CDS(cCDS_file, false);

        if (zero_coupon[zero_coupon.size()-1].first/365<T)
        {
                throw(std::logic_error("Maturity is longer than Zero Coupon Bond structure."));
        }
        if (myCDS[myCDS.size()-1].first<T || cCDS[cCDS.size()-1].first<T)
        {
                throw(std::logic_error("Maturity is longer than CDS term structure."));
        }
}

template<RateModels rate_model>
Cva<rate_model>::Cva(Type type_,
                     MC_Type mc_type_,
                     number T_,
                     number coupon_,
                     number a_,
                     number b_,
                     number sigma_,
                     number myR_,
                     number cR_,
                     std::string const & zc_file,
                     std::string const & myCDS_file,
                     std::string const & cCDS_file,
                     unsigned nsim_,
                     bool receiving_fix_,
                     unsigned max_gpus)
        :
        type(type_),
        mc_type(mc_type_),
        T(T_),
        coupon(coupon_),
        a(a_),
        b(b_),
        sigma(sigma_),
        myR(myR_),
        cR(cR_),
        nsim(nsim_),
        receiving_fix(receiving_fix_),
        number_of_gpu(std::min(max_gpus, tools_gpu::get_device_count())),
        simulation_grid(T-1),
        dt(simulation_grid+1, 1.),
        pricing_grid(simulation_grid+1),
        irs_CF(simulation_grid+1, coupon),
        EE(simulation_grid, 0.),
        NEE(simulation_grid, 0.),
        PFE(simulation_grid, 0.),
        NPFE(simulation_grid, 0.),
        EPE(0.),
        NEPE(0.),
        CVA(0.),
        time(0.),
        engine(rd()),
        n_dist(0.,1.),
        myCDS_interpolated(simulation_grid, 0.),
        cCDS_interpolated(simulation_grid, 0.),
        myDP(simulation_grid, 0.),
        cDP(simulation_grid, 0.),
        bt(),
        print_file(false)
{
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

        if (rate_model==CIR && mc_type==QMC)
        {
                std::cerr<<"Warning: Quasi Monte Carlo methods is not available for the CIR model. The program will use Monte Carlo instead.\n";
        }

        read_zcb(zc_file);
        irs_CF[irs_CF.size()-1]+=1.;
        read_CDS(myCDS_file, true);
        read_CDS(cCDS_file, false);

        if (zero_coupon[zero_coupon.size()-1].first/365<T)
        {
                throw(std::logic_error("Maturity is longer than Zero Coupon Bond structure."));
        }
        if (myCDS[myCDS.size()-1].first<T || cCDS[cCDS.size()-1].first<T)
        {
                throw(std::logic_error("Maturity is longer than CDS term structure."));
        }
}

template <RateModels rate_model>
void Cva<rate_model>::read_zcb(std::string const & zc_file)
{
        using namespace std;
        {
                ifstream read;
                read.open(zc_file.data());

                if (read.is_open())
                {
                        number temp1, temp2;

                        while (read>>temp1>>temp2)
                        {
                                zero_coupon.push_back(make_pair(temp1, temp2));
                        }
                }
                else
                {
                        throw(std::logic_error("Unable to open the ZeroCoupon file."));
                }

                read.close();

        }

        spot_rate=zero_coupon[0].second;
}

template <RateModels rate_model>
void Cva<rate_model>::read_CDS(std::string const & cds_file, bool mine)
{
        using namespace std;
        {
                ifstream read;
                read.open(cds_file.data());
                if (read.is_open())
                {
                        number temp1, temp2;

                        while (read>>temp1>>temp2)
                        {
                                if(mine)
                                {
                                        myCDS.push_back(make_pair(temp1, temp2));
                                }
                                else
                                {
                                        cCDS.push_back(make_pair(temp1, temp2));
                                }
                        }
                }
                else
                {
                        throw(std::logic_error("Unable to open the CDS file."));
                }
                read.close();
        }
        if (mine)
        {
                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        myCDS_interpolated[i]=interpolate(myCDS, i+1);
                        myDP[i]=1-exp(-myCDS_interpolated[i]*(i+1)/(10000*(1-myR)));
                }
        }
        else
        {
                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        cCDS_interpolated[i]=interpolate(cCDS, i+1);
                        cDP[i]=1-exp(-cCDS_interpolated[i]*(i+1)/(10000*(1-cR)));
                }
        }
}

template <RateModels rate_model>
void Cva<rate_model>::run()
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
        time=(real_e.tv_sec-real_s.tv_sec)*1.e3+(real_e.tv_usec - real_s.tv_usec)/1.e3;
}

template<>
number Cva<HullWhite>::linear_interpolation(std::vector<std::pair<number, number> > const & x, std::vector<number> const & y, number z)
{
        using namespace std;

        if (z<x[0].first || z>x[x.size()-1].first)
        {
                throw(logic_error("Cannot interpolate. The searched point is out of the intepolation domain."));
        }
        else if (x.size()!=y.size())
        {
                throw(logic_error("Data dimensions are inconsistent."));
        }
        else
        {
                unsigned pos=0;

                while (x[pos].first-z<0.)
                {
                        ++pos;
                }
                return y[pos-1]+(y[pos]-y[pos-1])*(z-x[pos-1].first)/(x[pos].first-x[pos-1].first);
        }
}

template <RateModels rate_model>
number Cva<rate_model>::interpolate(std::vector<std::pair<number, number> > const & x, number y)
{
        using namespace std;

        if (y<x[0].first || y>x[x.size()-1].first)
        {
                throw(logic_error("Cannot interpolate. The searched point is out of the intepolation domain."));
        }
        else
        {
                unsigned pos=0;

                while (x[pos].first-y<0.)
                {
                        ++pos;
                }
                return x[pos-1].second+(x[pos].second-x[pos-1].second)*(y-x[pos-1].first)/(x[pos].first-x[pos-1].first);
        }
}

//template<>
//number Cva<HullWhite>::zc_bondprices(number t)
template <RateModels rate_model>
number Cva<rate_model>::zc_bondprices(number t)
{

        unsigned pos=0;
        number basis=365;
        number price=1.;

        if (t>0.)
        {
                while (zero_coupon[pos].first-t*basis<0.)
                {
                        ++pos;
                }
                price=zero_coupon[pos-1].second+(zero_coupon[pos].second-zero_coupon[pos-1].second)*(t*basis-zero_coupon[pos-1].first)/(zero_coupon[pos].first-zero_coupon[pos-1].first);
        }

        return exp(-price*t);
}



template<>
number Cva<HullWhite>::alpha(number t)
{
        using namespace std;

        unsigned basis=365;

        vector<number> DF(zero_coupon.size());
        vector<number> DFi(zero_coupon.size());
        vector<number> delta_t(zero_coupon.size());
        vector<number> Fi(zero_coupon.size());

        for (unsigned i=0; i<zero_coupon.size(); ++i)
        {
                DF[i]=exp(-zero_coupon[i].first*zero_coupon[i].second/static_cast<number>(basis));
        }

        DFi[0]=DF[0];
        for (unsigned i=1; i<zero_coupon.size(); ++i)
        {
                DFi[i]=DF[i]/DF[i-1];
        }

        delta_t[0]=zero_coupon[0].first/static_cast<unsigned>(basis);
        for (unsigned i=1; i<zero_coupon.size(); ++i)
        {
                delta_t[i]=(zero_coupon[i].first-zero_coupon[i-1].first)/static_cast<number>(basis);
        }

        for (unsigned i=0; i<zero_coupon.size(); ++i)
        {
                Fi[i]=-log(DFi[i])/delta_t[i];
        }

        if (t>0.)
        {
                return linear_interpolation(zero_coupon, Fi, t*basis);
        }
        else
        {
                return zero_coupon[0].second;
        }

}

template<>
number Cva<HullWhite>::evaluate_B(number t, number T_)
{
        return (1.-exp(-a*(T_-t)))/a;
}

template<>
number Cva<HullWhite>::evaluate_A(number t, number T_)
{

        number B_t = zc_bondprices(t);
        number B_T = zc_bondprices(T_);
        number f_t = alpha(t);
        return exp(evaluate_B(t,T_)*f_t-sigma*sigma*(1.-exp(-2.*a*t))*evaluate_B(t,T_)*evaluate_B(t,T_)/(4.*a))*B_T/B_t;

}

template<>
number Cva<Vasicek>::evaluate_B(number t, number T_)
{
        return (1.-exp(-a*(T_-t)))/a;

}

template<>
number Cva<Vasicek>::evaluate_A(number t, number T_)
{
        number B=evaluate_B(t, T_);
        return (B-(T_-t))*(a*a*b-sigma*sigma/2.)/(a*a)-sigma*sigma*B*B/(4*a);

}

template<>
number Cva<CIR>::evaluate_B(number t, number T_)
{
        number h=sqrt(a*a+2*sigma*sigma);
        return (2*(exp(h*(T_-t))-1.))/(2*h+(a+h)*(exp(h*(T_-t))-1.));

}

template<>
number Cva<CIR>::evaluate_A(number t, number T_)
{
        number h=sqrt(a*a+2*sigma*sigma);
        number A=(2*h*exp((a+h)*(T_-t)/2.))/(2*h+(a+h)*(exp(h*(T_-t))-1.));
        return std::pow(A, 2*a*b/(sigma*sigma));

}

template <RateModels rate_model>
void Cva<rate_model>::ptfEval(std::vector<std::vector<number> > & MtM,
                              number t, std::vector<std::vector<number> > const & DF, unsigned nsim,
                              number c)
{
        for (unsigned i=0; i<DF[0].size(); ++i)
        {
                for (unsigned j=round(t); j<irs_CF.size(); ++j)
                {
                        MtM[round(t)-1][i]+=irs_CF[j]*DF[j-round(t)][i];
                }
        }
}

template<>
void Cva<HullWhite>::evaluateMtM(std::vector<std::vector<number> > & scen,
                                 std::vector<std::vector<number> > & MtM,
                                 std::ofstream & stream)
{
        using namespace std;
        using namespace statistics;

        vector<number> mean_add(simulation_grid);
        vector<number> mean_mult(simulation_grid);
        vector<number> sq_var(simulation_grid);

        for (unsigned i=0; i<simulation_grid; ++i)
        {
                pricing_grid[i]=i*dt[0];
                mean_add[i]=alpha(i+1)+sigma*sigma/(2.*a)*(1.-exp(-a*(i+1)))*(1.-exp(-a*(i+1)))
                            -(alpha(i)+sigma*sigma/(2.*a)*(1.-exp(-a*i))*(1.-exp(-a*i)))*exp(-a*dt[i]);
                mean_mult[i]=exp(-a*dt[i]);
                sq_var[i]=sqrt(sigma*sigma/(2.*a)*(1.-exp(-2.*a*dt[i])));
        }

        if (mc_type==MC)
        {
                scen=vector<vector<number> >(2, vector<number>(nsim));
        }
        else
        {
                scen=vector<vector<number> >(simulation_grid, vector<number>(nsim));

                vector<number> data(simulation_grid*nsim);

                curandGenerator_t gen;
                // Create pseudo-random number generator
                curandCreateGeneratorHost(&gen, CURAND_RNG_QUASI_SOBOL64);
                // Set seed
                curandSetQuasiRandomGeneratorDimensions(gen, simulation_grid);
                // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                curandGenerateNormal(gen, data.data(), nsim*simulation_grid, 0., 1.);
#else
                curandGenerateNormalDouble(gen, data.data(), nsim*simulation_grid, 0., 1.);
#endif
                curandDestroyGenerator(gen);

                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                scen[i][j]=data[i*nsim+j];
                        }
                }
        }

        vector<number> A(simulation_grid), B(simulation_grid);

        if (print_file)
        {
                std::string name("matlab/HW");
                name.append(std::to_string(this->id));
                name.append(".m");
                stream.open(name);
                if (!stream.is_open())
                {
                        throw(std::logic_error("Cannot open a file."));
                }
                stream<<"r=[ ";
        }

        for (unsigned i=0; i<simulation_grid; ++i)
        {
                if (i==0)
                {
                        if (mc_type==MC)
                        {
                                for (unsigned j=0; j<nsim; ++j)
                                {
                                        scen[i][j]=spot_rate*mean_mult[i]+mean_add[i]+sq_var[i]*n_dist(engine);
                                        if (print_file)
                                        {
                                                if (j!=nsim-1)
                                                {
                                                        stream<<scen[i][j]<<", ";
                                                }
                                                else
                                                {
                                                        stream<<scen[i][j]<<"; ";
                                                }
                                        }
                                }
                        }
                        else
                        {
                                for (unsigned j=0; j<nsim; ++j)
                                {
                                        scen[i][j]=spot_rate*mean_mult[i]+mean_add[i]+sq_var[i]*scen[i][j];
                                        if (print_file)
                                        {
                                                if (j!=nsim-1)
                                                {
                                                        stream<<scen[i][j]<<", ";
                                                }
                                                else
                                                {
                                                        stream<<scen[i][j]<<"; ";
                                                }
                                        }
                                }
                        }
                }
                else
                {
                        if (mc_type==MC)
                        {
                                for (unsigned j=0; j<nsim; ++j)
                                {
                                        scen[i%2][j]=scen[(i-1)%2][j]*mean_mult[i]+mean_add[i]+sq_var[i]*n_dist(engine);
                                        if (print_file)
                                        {
                                                if (j!=nsim-1)
                                                {
                                                        stream<<scen[i%2][j]<<", ";
                                                }
                                                else if (j==nsim-1 && i==simulation_grid-1)
                                                {
                                                        stream<<scen[i%2][j]<<" ];\n";
                                                }
                                                else
                                                {
                                                        stream<<scen[i%2][j]<<"; ";
                                                }
                                        }
                                }
                        }
                        else
                        {
                                for (unsigned j=0; j<nsim; ++j)
                                {
                                        scen[i][j]=scen[i-1][j]*mean_mult[i]+mean_add[i]+sq_var[i]*scen[i][j];
                                        if (print_file)
                                        {
                                                if (j!=nsim-1)
                                                {
                                                        stream<<scen[i%2][j]<<", ";
                                                }
                                                else if (j==nsim-1 && i==simulation_grid-1)
                                                {
                                                        stream<<scen[i%2][j]<<" ];\n";
                                                }
                                                else
                                                {
                                                        stream<<scen[i%2][j]<<"; ";
                                                }
                                        }
                                }
                        }
                }

                pricing_grid.pop_back();
                A.clear();
                A.resize(pricing_grid.size());
                B.clear();
                B.resize(pricing_grid.size());

                for (unsigned j=0; j<A.size(); ++j)
                {
                        A[j]=evaluate_A(static_cast<number>(i+1), static_cast<number>(j+i+2));
                        B[j]=evaluate_B(static_cast<number>(i+1), static_cast<number>(j+i+2));
                }

                vector<vector<number> > DF_vec (pricing_grid.size(), vector<number>(nsim));

                if (mc_type==MC)
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                for (unsigned k=0; k<pricing_grid.size(); ++k)
                                {
                                        DF_vec[k][j]=exp(-scen[i%2][j]*B[k])*A[k];
                                }
                        }
                }
                else
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                for (unsigned k=0; k<pricing_grid.size(); ++k)
                                {
                                        DF_vec[k][j]=exp(-scen[i][j]*B[k])*A[k];
                                }
                        }
                }


                ptfEval(MtM, static_cast<number>(i+1), DF_vec, nsim, coupon);
        }
}

template<>
void Cva<Vasicek>::evaluateMtM(std::vector<std::vector<number> > & scen,
                               std::vector<std::vector<number> > & MtM,
                               std::ofstream & stream)
{
        using namespace std;
        using namespace statistics;

        number mean_add=b*(1-exp(-a*dt[0]));
        number mean_mult=exp(-a*dt[0]);
        number sq_var=sigma*sqrt((1-exp(-2*a*dt[0]))/(2*a));

        for (unsigned i=0; i<simulation_grid; ++i)
        {
                pricing_grid[i]=i*dt[0];
        }

        if (mc_type==MC)
        {
                scen=vector<vector<number> >(2, vector<number>(nsim));
        }
        else
        {
                scen=vector<vector<number> >(simulation_grid, vector<number>(nsim));

                vector<number> data(simulation_grid*nsim);

                curandGenerator_t gen;
                // Create pseudo-random number generator
                curandCreateGeneratorHost(&gen, CURAND_RNG_QUASI_SOBOL64);
                // Set seed
                curandSetQuasiRandomGeneratorDimensions(gen, simulation_grid);
                // Generate x and y on device
#ifdef __SINGLE_PRECISION__
                curandGenerateNormal(gen, data.data(), nsim*simulation_grid, 0., 1.);
#else
                curandGenerateNormalDouble(gen, data.data(), nsim*simulation_grid, 0., 1.);
#endif
                curandDestroyGenerator(gen);

                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                scen[i][j]=data[i*nsim+j];
                        }
                }
        }

        vector<number> A(simulation_grid), B(simulation_grid);

        if (print_file)
        {
                std::string name("matlab/V");
                name.append(std::to_string(this->id));
                name.append(".m");
                stream.open(name);

                if (!stream.is_open())
                {
                        throw(std::logic_error("Cannot open a file."));
                }
                stream<<"r=[ ";
        }
        for (unsigned i=0; i<simulation_grid; ++i)
        {
                if (i==0)
                {
                        if (mc_type==MC)
                        {
                                for (unsigned j=0; j<nsim; ++j)
                                {
                                        scen[i][j]=spot_rate*mean_mult+mean_add+sq_var*n_dist(engine);
                                        if (print_file)
                                        {
                                                if (j!=nsim-1)
                                                {
                                                        stream<<scen[i][j]<<", ";
                                                }
                                                else
                                                {
                                                        stream<<scen[i][j]<<"; ";
                                                }
                                        }
                                }
                        }
                        else
                        {
                                for (unsigned j=0; j<nsim; ++j)
                                {
                                        scen[i][j]=spot_rate*mean_mult+mean_add+sq_var*scen[i][j];
                                        if (print_file)
                                        {
                                                if (j!=nsim-1)
                                                {
                                                        stream<<scen[i][j]<<", ";
                                                }
                                                else
                                                {
                                                        stream<<scen[i][j]<<"; ";
                                                }
                                        }
                                }
                        }
                }
                else
                {
                        if (mc_type==MC)
                        {
                                for (unsigned j=0; j<nsim; ++j)
                                {
                                        scen[i%2][j]=scen[(i-1)%2][j]*mean_mult+mean_add+sq_var*n_dist(engine);
                                        if (print_file)
                                        {
                                                if (j!=nsim-1)
                                                {
                                                        stream<<scen[i%2][j]<<", ";
                                                }
                                                else if (j==nsim-1 && i==simulation_grid-1)
                                                {
                                                        stream<<scen[i%2][j]<<" ];\n";
                                                }
                                                else
                                                {
                                                        stream<<scen[i%2][j]<<"; ";
                                                }
                                        }
                                }
                        }
                        else
                        {
                                for (unsigned j=0; j<nsim; ++j)
                                {
                                        scen[i][j]=scen[i-1][j]*mean_mult+mean_add+sq_var*scen[i][j];
                                        if (print_file)
                                        {
                                                if (j!=nsim-1)
                                                {
                                                        stream<<scen[i%2][j]<<", ";
                                                }
                                                else if (j==nsim-1 && i==simulation_grid-1)
                                                {
                                                        stream<<scen[i%2][j]<<" ];\n";
                                                }
                                                else
                                                {
                                                        stream<<scen[i%2][j]<<"; ";
                                                }
                                        }
                                }
                        }
                }

                pricing_grid.pop_back();
                A.clear();
                A.resize(pricing_grid.size());
                B.clear();
                B.resize(pricing_grid.size());

                //cout<<"*** "<<i<<" ***\n";
                for (unsigned j=0; j<A.size(); ++j)
                {
                        A[j]=evaluate_A(static_cast<number>(i+1), static_cast<number>(j+i+2));
                        B[j]=evaluate_B(static_cast<number>(i+1), static_cast<number>(j+i+2));
                        //cout<<"A: "<<A[j]<<" B: "<<B[j]<<"\t";
                }
                //cout<<"\n";

                vector<vector<number> > DF_vec (pricing_grid.size(), vector<number>(nsim));

                if (mc_type==MC)
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                for (unsigned k=0; k<pricing_grid.size(); ++k)
                                {
                                        DF_vec[k][j]=exp(-scen[i%2][j]*B[k]+A[k]);
                                }
                        }
                }
                else
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                for (unsigned k=0; k<pricing_grid.size(); ++k)
                                {
                                        DF_vec[k][j]=exp(-scen[i][j]*B[k]+A[k]);
                                }
                        }
                }

                ptfEval(MtM, static_cast<number>(i+1), DF_vec, nsim, coupon);
        }
}

template<>
void Cva<CIR>::evaluateMtM(std::vector<std::vector<number> > & scen,
                           std::vector<std::vector<number> > & MtM,
                           std::ofstream & stream)
{
        using namespace std;
        using namespace statistics;

        scen=vector<vector<number> >(2, vector<number>(nsim));

        vector<number> mean_add(simulation_grid);
        vector<number> mean_mult(simulation_grid);
        vector<number> sq_var(simulation_grid);

        for (unsigned i=0; i<simulation_grid; ++i)
        {
                pricing_grid[i]=i*dt[0];
                mean_mult[i]=exp(-a*dt[i]);
                sq_var[i]=sqrt(sigma*sigma/(2.*a)*(1.-exp(-2.*a*dt[i])));
        }

        number v = sigma*sigma;
        number d = 4*a*b/v;

        vector<number> e(simulation_grid);
        vector<number> c(simulation_grid);
        for (unsigned i=0; i<simulation_grid; ++i)
        {
                e[i]=exp(-a*dt[i]);
                c[i]=v*(1-e[i])/(4*a);
        }

        number l=0;
        number random=0;

        std::mt19937				engine2(rd());
        std::uniform_real_distribution<>	u_dist(0.,1.);

        vector<number> A(simulation_grid), B(simulation_grid);

        if (print_file)
        {
                std::string name("matlab/CIR");
                name.append(std::to_string(this->id));
                name.append(".m");
                stream.open(name);

                if (!stream.is_open())
                {
                        throw(std::logic_error("Cannot open a file."));
                }
                stream<<"r=[ ";
        }
        for (unsigned i=0; i<simulation_grid; ++i)
        {
                if (i==0)
                {
                        l=spot_rate*e[i]/c[i];
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                random=n_dist(engine);
                                number chi=GetChiSquare(engine, n_dist, engine2, u_dist, d-1.);
                                random=(random+sqrt(l))*(random+sqrt(l))+chi;
                                scen[i][j]=c[i]*random;
                                if (print_file)
                                {
                                        if (j!=nsim-1)
                                        {
                                                stream<<scen[i][j]<<", ";
                                        }
                                        else
                                        {
                                                stream<<scen[i][j]<<"; ";
                                        }
                                }
                        }
                }
                else
                {
                        for (unsigned j=0; j<nsim; ++j)
                        {
                                l=scen[(i-1)%2][j]*e[i]/c[i];
                                random=n_dist(engine);
                                number chi=GetChiSquare(engine, n_dist, engine2, u_dist, d-1.);
                                random=(random+sqrt(l))*(random+sqrt(l))+chi;
                                scen[i%2][j]=c[i]*random;
                                if (print_file)
                                {
                                        if (j!=nsim-1)
                                        {
                                                stream<<scen[i%2][j]<<", ";
                                        }
                                        else if (j==nsim-1 && i==simulation_grid-1)
                                        {
                                                stream<<scen[i%2][j]<<" ];\n";
                                        }
                                        else
                                        {
                                                stream<<scen[i%2][j]<<"; ";
                                        }
                                }
                        }
                }

                pricing_grid.pop_back();
                A.clear();
                A.resize(pricing_grid.size());
                B.clear();
                B.resize(pricing_grid.size());

                for (unsigned j=0; j<A.size(); ++j)
                {
                        A[j]=evaluate_A(static_cast<number>(i+1), static_cast<number>(j+i+2));
                        B[j]=evaluate_B(static_cast<number>(i+1), static_cast<number>(j+i+2));
                }

                vector<vector<number> > DF_vec (pricing_grid.size(), vector<number>(nsim));

                for (unsigned j=0; j<nsim; ++j)
                {
                        for (unsigned k=0; k<pricing_grid.size(); ++k)
                        {
                                DF_vec[k][j]=exp(-scen[i%2][j]*B[k])*A[k];
                        }
                }

                ptfEval(MtM, static_cast<number>(i+1), DF_vec, nsim, coupon);
        }
}

template<RateModels rate_model>
void Cva<rate_model>::evaluate_cva()
{
        number DF=0.;

        for (unsigned i=0; i<simulation_grid; ++i)
        {
                DF=zc_bondprices(i+1);
                if (i==0)
                {
                        CVA+=(1-cR)*DF*EE[i]*cDP[i]-(1-myR)*DF*NEE[i]*myDP[i];
                }
                else
                {
                        CVA+=(1-cR)*DF*EE[i]*(1-myDP[i-1])*(cDP[i]-cDP[i-1])
                             -(1-myR)*DF*NEE[i]*(1-cDP[i-1])*(myDP[i]-myDP[i-1]);
                }
        }
}

template<RateModels rate_model>
void Cva<rate_model>::run_cpu()
{
        using namespace std;
        using namespace statistics;

        vector<vector<number> > scen;
        std::vector<std::vector<number> > MtM(simulation_grid, std::vector<number>(nsim, -1.));

        std::ofstream stream;

        evaluateMtM(scen, MtM, stream);

        if (print_file)
        {
                stream<<"MtM=[";
                for (unsigned i=0; i<MtM.size(); ++i)
                {
                        for (unsigned j=0; j<MtM[0].size(); ++j)
                        {
                                if (j!=MtM[0].size()-1)
                                {
                                        stream<<(receiving_fix?MtM[i][j]:-MtM[i][j])<<", ";
                                }
                                else if (j==MtM[0].size()-1 && i==MtM.size()-1)
                                {
                                        stream<<(receiving_fix?MtM[i][j]:-MtM[i][j])<<" ];\n";
                                }
                                else
                                {
                                        stream<<(receiving_fix?MtM[i][j]:-MtM[i][j])<<"; ";
                                }
                        }
                }
        }
        for (unsigned i=0; i<EE.size(); ++i)
        {
                for (unsigned j=0; j<nsim; ++j)
                {
                        EE[i]+=max(MtM[i][j], static_cast<number>(0.))/nsim;
                        NEE[i]+=max(-MtM[i][j], static_cast<number>(0.))/nsim;
                }
                EPE+=EE[i]/EE.size();
                NEPE+=NEE[i]/EE.size();
        }

        for (unsigned i=0; i<MtM.size(); ++i)
        {
                sort(MtM[i].begin(), MtM[i].end());
        }

        for (unsigned i=0; i<PFE.size(); ++i)
        {
                if (round(Constants::quantile*nsim)<nsim)
                {
                        PFE[i]=max(static_cast<number>(0.),MtM[i][round(Constants::quantile*nsim)]);
                }
                else
                {
                        PFE[i]=max(static_cast<number>(0.),MtM[i][round(Constants::quantile*nsim)-1]);
                }
                NPFE[i]=max(static_cast<number>(0.), -MtM[i][round((1.-Constants::quantile)*nsim)]);
        }

        if (!receiving_fix)
        {
                swap(EE, NEE);
                swap(PFE, NPFE);
                swap(EPE, NEPE);
        }

        evaluate_cva();

        if (print_file)
        {
                stream<<"EE = [ 0, ";
                for (unsigned i=0; i<EE.size(); ++i)
                {
                        if (i<EE.size()-1)
                        {
                                stream<<EE[i]<<", ";
                        }
                        else
                        {
                                stream<<EE[i]<<", 0];\n";
                        }
                }
                stream<<"NEE = [ 0, ";
                for (unsigned i=0; i<NEE.size(); ++i)
                {
                        if (i<EE.size()-1)
                        {
                                stream<<NEE[i]<<", ";
                        }
                        else
                        {
                                stream<<NEE[i]<<", 0];\n";
                        }
                }
                stream<<"PFE = [ 0, ";
                for (unsigned i=0; i<PFE.size(); ++i)
                {
                        if (i<EE.size()-1)
                        {
                                stream<<PFE[i]<<", ";
                        }
                        else
                        {
                                stream<<PFE[i]<<", 0];\n";
                        }
                }
                stream<<"NPFE = [ 0, ";
                for (unsigned i=0; i<PFE.size(); ++i)
                {
                        if (i<EE.size()-1)
                        {
                                stream<<NPFE[i]<<", ";
                        }
                        else
                        {
                                stream<<NPFE[i]<<", 0];\n";
                        }
                }
                stream<<"EPE = "<<EPE<<";\n";
                stream<<"NEPE = "<<NEPE<<";\n";
                stream<<"CVA = "<<CVA<<";\n";

                stream.close();
        }
}

template<RateModels rate_model>
void Cva<rate_model>::run_gpu()
{
        using namespace std;

        vector<number> data;

        if (rate_model==HullWhite)
        {

                vector<number> mean_add(simulation_grid);
                vector<number> mean_mult(simulation_grid);
                vector<number> sq_var(simulation_grid);

                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        pricing_grid[i]=i*dt[0];
                        mean_add[i]=alpha(i+1)+sigma*sigma/(2.*a)*(1.-exp(-a*(i+1)))*(1.-exp(-a*(i+1)))
                                    -(alpha(i)+sigma*sigma/(2.*a)*(1.-exp(-a*i))*(1.-exp(-a*i)))*exp(-a*dt[i]);
                        mean_mult[i]=exp(-a*dt[i]);
                        sq_var[i]=sqrt(sigma*sigma/(2.*a)*(1.-exp(-2.*a*dt[i])));
                }

                data.reserve(mean_add.size()+mean_mult.size()+sq_var.size());
                data.insert(data.end(), mean_add.begin(), mean_add.end());
                data.insert(data.end(), mean_mult.begin(), mean_mult.end());
                data.insert(data.end(), sq_var.begin(), sq_var.end());
                data.push_back(a);
                data.push_back(sigma);

        }
        else if (rate_model==Vasicek)
        {

                number mean_add=b*(1-exp(-a*dt[0]));
                number mean_mult=exp(-a*dt[0]);
                number sq_var=sigma*sqrt((1-exp(-2*a*dt[0]))/(2*a));

                data.push_back(mean_add);
                data.push_back(mean_mult);
                data.push_back(sq_var);
                data.push_back(a);
                data.push_back(b);
                data.push_back(sigma);

        }
        else if (rate_model==CIR)
        {

                if (2*a*b<sigma*sigma)
                {
                        throw(logic_error("2ab<sigma^2, cannot continue."));
                }

                number v = sigma*sigma;
                number d = 4*a*b/v;

                vector<number> e(simulation_grid);
                vector<number> c(simulation_grid);
                for (unsigned i=0; i<simulation_grid; ++i)
                {
                        e[i]=exp(-a*dt[i]);
                        c[i]=v*(1-e[i])/(4*a);
                }

                data.reserve(c.size()+e.size());
                data.insert(data.end(), c.begin(), c.end());
                data.insert(data.end(), e.begin(), e.end());
                data.push_back(v);
                data.push_back(d);
                data.push_back(a);
                data.push_back(b);
                data.push_back(sigma);

        }

        vector<number> A(simulation_grid*simulation_grid, 0.);
        vector<number> B(simulation_grid*simulation_grid, 0.);

        for (unsigned i=0; i<simulation_grid; ++i)
        {
                for (unsigned j=0; j<simulation_grid-(i); ++j)
                {
                        A[i*simulation_grid+j]=evaluate_A(static_cast<number>(i+1), static_cast<number>(j+i+2));
                        B[i*simulation_grid+j]=evaluate_B(static_cast<number>(i+1), static_cast<number>(j+i+2));
                }
        }

        cva_gpu::evaluate_cva(rate_model, mc_type, nsim/number_of_gpu, simulation_grid, A, B,
                              pricing_grid, data, zero_coupon, irs_CF,
                              bt, number_of_gpu, EPE, NEPE, EE, NEE, PFE, NPFE, rd());

        if (!receiving_fix)
        {
                swap(EE, NEE);
                swap(PFE, NPFE);
                swap(EPE, NEPE);
        }

        evaluate_cva();
}

template <RateModels rate_model>
std::ostream & operator<< (std::ostream & out, const Cva<rate_model> & cva)
{
        out<<"Results for object "<<cva.id<<":\n";
        out<<"EPE = "<<cva.EPE<<"\n";
        out<<"NEPE = "<<cva.NEPE<<"\n";
        out<<"EE = ";
        for (unsigned i=0; i<cva.EE.size(); ++i)
        {
                out<<cva.EE[i]<<" ";
        }
        out<<"\nNEE = ";
        for (unsigned i=0; i<cva.EE.size(); ++i)
        {
                out<<cva.NEE[i]<<" ";
        }
        out<<"\nPFE = ";
        for (unsigned i=0; i<cva.PFE.size(); ++i)
        {
                out<<cva.PFE[i]<<" ";
        }
        out<<"\nNPFE = ";
        for (unsigned i=0; i<cva.PFE.size(); ++i)
        {
                out<<cva.NPFE[i]<<" ";
        }
        out<<"\nCVA = "<<cva.CVA;
        out<<"\nTime elapsed = " << cva.time <<"ms.\n";
        out << "******\n";

        return out;
}

#endif
