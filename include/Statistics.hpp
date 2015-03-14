#ifndef __Statistics_hpp
#define __Statistics_hpp

#include <cmath>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <random>

#include "PrecisionTypes.hpp"

namespace statistics
{

//! This function evaluates the cdf of the normal
/*! Code found at http://www.johndcook.com/stand_alone_code.html .
 * All code here is in the public domain. Do whatever you want with it, no strings attached.
 */
number phi(number x);

//! This function generates a gamma random variable
/*! Code found at http://www.johndcook.com/stand_alone_code.html
 *  All code here is in the public domain. Do whatever you want with it, no strings attached.
 */
number GetGamma(std::mt19937 & e1, std::normal_distribution<number> & n_dist,
                std::mt19937 & e2, std::uniform_real_distribution<number> & u_dist, number shape, number scale);
//! This function generates a chi squared with non-integer degrees of freedom
/*! Code found at http://www.johndcook.com/stand_alone_code.html
 *  All code here is in the public domain. Do whatever you want with it, no strings attached.
 */
number GetChiSquare(std::mt19937 & e1, std::normal_distribution<number> & n_dist,
                    std::mt19937 & e2, std::uniform_real_distribution<number> & u_dist, number degreesOfFreedom);

//! This function evaluates the icdf of the gaussian distribution
/*! Code fount at http://people.sc.fsu.edu/~jburkardt/f_src/asa243/asa243.html
 */
number alnorm ( number x, bool upper );

//! This function calculates the beta incomplete function
/*! Code fount at http://people.sc.fsu.edu/~jburkardt/f_src/asa243/asa243.html
 */
number betain ( number x, number p, number q, number beta, int * ifault );

//! This function calculates the cdf of a Student's t distribution
/*! Code fount at http://people.sc.fsu.edu/~jburkardt/f_src/asa243/asa243.html
 */
number tnc ( number t, number df, number delta, int * ifault );

}

#endif