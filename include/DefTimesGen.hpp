#ifndef __DefTimesGen_hpp
#define __DefTimesGen_hpp

#include <iostream>
#include <fstream>
#include <time.h>
#include <iomanip>
#include <vector>
#include <ctime>
#include <sys/time.h>
#include <memory>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <curand.h>

#include "tools.hpp"
#include "EigenIterator.hpp"
#include "CopulaType.hpp"
#include "DefTimes.cuh"
#include "Statistics.hpp"
#include "Constants.hpp"

//! Abstract base class for generating the default times according to a copula.
/*! This class defines the layout and implements the common methods of the DefTimesGen-type classes, which generate the default times according to a Gaussian of a Student's t copula.
 */
class DefTimesGen
{
public:
        //! Destructor
        virtual ~DefTimesGen();
        //! Constructor for CPU
        DefTimesGen(Type, MC_Type, unsigned, number, number);
        //! Constructor for GPU
        DefTimesGen(Type, MC_Type, unsigned, number, number, unsigned);

        //!
        /*! A function that allows the user to set the maximum number of threads per block.
         */
        virtual void set_max_threads_per_block(unsigned);

        //!
        /*! This method generates the default times on the CPU
         */
        virtual std::unique_ptr<tools::Matrix> generate_deftimes_CPU(unsigned)=0;
        //!
        /*! This method generates the default times on the GPU
         */
        virtual std::vector<number *> generate_deftimes_GPU(unsigned)=0;

protected:

        Type            type;
        MC_Type         mc_type;

        unsigned        dim;
        number          rho;
        number          lambda;
        tools::Matrix   S;

        std::unique_ptr<tools::Matrix>	ptr;
        std::vector<number *>		d_T;

        unsigned                        number_of_gpu;

        std::random_device		rd;
        std::mt19937			engine;
        std::normal_distribution<>	n_dist;

        tools::Block_Threads		bt;

        virtual void compute_copula_matrix();

};

//! Class for generating default times according to a Gaussian Copula
/*! This class inherits from DefTimesGen and implements the methods which generate default times on CPU and GPU
 */
class DefTimesGen_GaussianCopula: public DefTimesGen
{
public:
        //! Constructor for CPU
        /*! This is the constructor of the class DefTimesGen_tCopula. The argument list is the following:
         * \param type_         type of calculation: CPU or GPU
         * \param mc_type_      type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
         * \param dim_          dimension of the reference portfolio
         * \param rho_          correlation index in the reference portfolio. This index must be in the following set:\f[ \rho\in\left(\frac{-1}{dim-1},1\right).\f]
         * \param lambda_       hazard rate
         */
        DefTimesGen_GaussianCopula(Type type_,
                                   MC_Type mc_type_,
                                   unsigned dim_,
                                   number rho_,
                                   number lambda_);
        //! Constructor for GPU
        /*! This is the constructor of the class DefTimesGen_tCopula. The argument list is the following:
         * \param type_         type of calculation: CPU or GPU
         * \param mc_type_      type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
         * \param dim_          dimension of the reference portfolio
         * \param rho_          correlation index in the reference portfolio. This index must be in the following set:\f[ \rho\in\left(\frac{-1}{dim-1},1\right).\f]
         * \param lambda_       hazard rate
         * \param num_gpus      number of gpus available
         */
        DefTimesGen_GaussianCopula(Type type_,
                                   MC_Type mc_type_,
                                   unsigned dim_,
                                   number rho_,
                                   number lambda_,
                                   unsigned num_gpus);

        //!
        /*! This method generates the default times on the CPU
         */
        virtual std::unique_ptr<tools::Matrix> generate_deftimes_CPU(unsigned);
        //!
        /*! This method generates the default times on the GPU
         */
        virtual std::vector<number *> generate_deftimes_GPU(unsigned);

};

//! Class for generating default times according to a Student's t Copula
/*! This class inherits from DefTimesGen and implements the methods which generate default times on CPU and GPU
 */
class DefTimesGen_tCopula: public DefTimesGen
{
protected:
        unsigned dof;
        std::mt19937			engine2;
        std::chi_squared_distribution<>	chi_dist;

public:
        //! Constructor for CPU
        /*! This is the constructor of the class DefTimesGen_tCopula. The argument list is the following:
         * \param type_         type of calculation: CPU or GPU
         * \param mc_type_      type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
         * \param dim_          dimension of the reference portfolio
         * \param rho_          correlation index in the reference portfolio. This index must be in the following set:\f[ \rho\in\left(\frac{-1}{dim-1},1\right).\f]
         * \param lambda_       hazard rate
         * \param dof_          degrees of freedom of the Student's t copula.
         */
        DefTimesGen_tCopula(Type type_,
                            MC_Type mc_type_,
                            unsigned dim_,
                            number rho_,
                            number lambda_,
                            unsigned dof_);
        //! Constructor for GPU
        /*! This is the constructor of the class DefTimesGen_tCopula. The argument list is the following:
         * \param type_         type of calculation: CPU or GPU
         * \param mc_type_      type of algorithm: MC for Monte Carlo, QMC for Quasi Monte Carlo (Sobol)
         * \param dim_          dimension of the reference portfolio
         * \param rho_          correlation index in the reference portfolio. This index must be in the following set:\f[ \rho\in\left(\frac{-1}{dim-1},1\right).\f]
         * \param lambda_       hazard rate
         * \param num_gpus      number of gpus available
         * \param dof_          degrees of freedom of the Student's t copula.
         */
        DefTimesGen_tCopula(Type type_,
                            MC_Type mc_type_,
                            unsigned dim_,
                            number rho_,
                            number lambda_,
                            unsigned num_gpus,
                            unsigned dof_);

        //!
        /*! This method generates the default times on the CPU
         */
        virtual std::unique_ptr<tools::Matrix> generate_deftimes_CPU(unsigned);
        //!
        /*! This method generates the default times on the GPU
         */
        virtual std::vector<number *> generate_deftimes_GPU(unsigned);
};

#endif
