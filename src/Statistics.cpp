#include "Statistics.hpp"

namespace statistics
{

// Code found at http://www.johndcook.com/stand_alone_code.html .
// All code here is in the public domain. Do whatever you want with it, no strings attached.
number phi(number x)
{
        // constants
        number a1 =  0.254829592;
        number a2 = -0.284496736;
        number a3 =  1.421413741;
        number a4 = -1.453152027;
        number a5 =  1.061405429;
        number p  =  0.3275911;

        // Save the sign of x
        int sign = 1;
        if (x < 0)
        {
                sign = -1;
        }
        x = fabs(x)/sqrt(2.0);

        // A&S formula 7.1.26
        number t = 1.0/(1.0 + p*x);
        number y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

        return 0.5*(1.0 + sign*y);
}

// Code found at http://www.johndcook.com/stand_alone_code.html .
// All code here is in the public domain. Do whatever you want with it, no strings attached.
number GetGamma(std::mt19937 & e1, std::normal_distribution<number> & n_dist,
                std::mt19937 & e2, std::uniform_real_distribution<number> & u_dist, number shape, number scale)
{
        // Implementation based on "A Simple Method for Generating Gamma Variables"
        // by George Marsaglia and Wai Wan Tsang.  ACM Transactions on Mathematical Software
        // Vol 26, No 3, September 2000, pages 363-372.

        number d, c, x, xsquared, v, u;

        if (shape >= 1.0)
        {
                d = shape - 1.0/3.0;
                c = 1.0/sqrt(9.0*d);
                for (;;)
                {
                        do
                        {
                                x = n_dist(e1);
                                v = 1.0 + c*x;
                        }
                        while (v <= 0.0);
                        v = v*v*v;
                        u = u_dist(e2);
                        xsquared = x*x;
                        if (u < 1.0 -.0331*xsquared*xsquared || log(u) < 0.5*xsquared + d*(1.0 - v + log(v)))
                        {
                                return scale*d*v;
                        }
                }
        }
        else if (shape <= 0.0)
        {
                std::stringstream os;
                os << "Shape parameter must be positive." << "\n"
                   << "Received shape parameter " << shape;
                throw std::invalid_argument( os.str() );
        }
        else
        {
                number g = GetGamma(e1, n_dist, e2, u_dist, shape+1.0, 1.0);
                number w = u_dist(e2);
                return scale*g*pow(w, 1.0/shape);
        }
}

// Code found at http://www.johndcook.com/stand_alone_code.html .
// All code here is in the public domain. Do whatever you want with it, no strings attached.
number GetChiSquare(std::mt19937 & e1, std::normal_distribution<number> & n_dist,
                    std::mt19937 & e2, std::uniform_real_distribution<number> & u_dist, number degreesOfFreedom)
{
        // A chi squared distribution with n degrees of freedom
        // is a gamma distribution with shape n/2 and scale 2.
        return GetGamma(e1, n_dist, e2, u_dist, 0.5 * degreesOfFreedom, 2.0);
}

//****************************************************************************80

number alnorm ( number x, bool upper )

//****************************************************************************80
//
//  Purpose:
//
//    ALNORM computes the cumulative density of the standard normal distribution.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 January 2008
//
//  Author:
//
//    Original FORTRAN77 version by David Hill.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    David Hill,
//    Algorithm AS 66:
//    The Normal Integral,
//    Applied Statistics,
//    Volume 22, Number 3, 1973, pages 424-427.
//
//  Parameters:
//
//    Input, number X, is one endpoint of the semi-infinite interval
//    over which the integration takes place.
//
//    Input, bool UPPER, determines whether the upper or lower
//    interval is to be integrated:
//    .TRUE.  => integrate from X to + Infinity;
//    .FALSE. => integrate from - Infinity to X.
//
//    Output, number ALNORM, the integral of the standard normal
//    distribution over the desired interval.
//
{
        number a1 = 5.75885480458;
        number a2 = 2.62433121679;
        number a3 = 5.92885724438;
        number b1 = -29.8213557807;
        number b2 = 48.6959930692;
        number c1 = -0.000000038052;
        number c2 = 0.000398064794;
        number c3 = -0.151679116635;
        number c4 = 4.8385912808;
        number c5 = 0.742380924027;
        number c6 = 3.99019417011;
        number con = 1.28;
        number d1 = 1.00000615302;
        number d2 = 1.98615381364;
        number d3 = 5.29330324926;
        number d4 = -15.1508972451;
        number d5 = 30.789933034;
        number ltone = 7.0;
        number p = 0.398942280444;
        number q = 0.39990348504;
        number r = 0.398942280385;
        bool up;
        number utzero = 18.66;
        number value;
        number y;
        number z;

        up = upper;
        z = x;

        if ( z < 0.0 )
        {
                up = !up;
                z = - z;
        }

        if ( ltone < z && ( ( !up ) || utzero < z ) )
        {
                if ( up )
                {
                        value = 0.0;
                }
                else
                {
                        value = 1.0;
                }
                return value;
        }

        y = 0.5 * z * z;

        if ( z <= con )
        {
                value = 0.5 - z * ( p - q * y
                                    / ( y + a1 + b1
                                        / ( y + a2 + b2
                                            / ( y + a3 ))));
        }
        else
        {
                value = r * exp ( - y )
                        / ( z + c1 + d1
                            / ( z + c2 + d2
                                / ( z + c3 + d3
                                    / ( z + c4 + d4
                                        / ( z + c5 + d5
                                            / ( z + c6 ))))));
        }

        if ( !up )
        {
                value = 1.0 - value;
        }

        return value;
}
//****************************************************************************80

number betain ( number x, number p, number q, number beta, int * ifault )

//****************************************************************************80
//
//  Purpose:
//
//    BETAIN computes the incomplete Beta function ratio.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    23 January 2008
//
//  Author:
//
//    Original FORTRAN77 version by KL Majumder, GP Bhattacharjee.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    KL Majumder, GP Bhattacharjee,
//    Algorithm AS 63:
//    The incomplete Beta Integral,
//    Applied Statistics,
//    Volume 22, Number 3, 1973, pages 409-411.
//
//  Parameters:
//
//    Input, number X, the argument, between 0 and 1.
//
//    Input, number P, Q, the parameters, which
//    must be positive.
//
//    Input, number BETA, the logarithm of the complete
//    beta function.
//
//    Output, int *IFAULT, error flag.
//    0, no error.
//    nonzero, an error occurred.
//
//    Output, number BETAIN, the value of the incomplete
//    Beta function ratio.
//
{
        number acu = 0.1E-14;
        number ai;
        //number betain;
        number cx;
        bool indx;
        int ns;
        number pp;
        number psq;
        number qq;
        number rx;
        number temp;
        number term;
        number value;
        number xx;

        value = x;
        *ifault = 0;
        //
        //  Check the input arguments.
        //
        if ( p <= 0.0 || q <= 0.0 )
        {
                *ifault = 1;
                return value;
        }

        if ( x < 0.0 || 1.0 < x )
        {
                *ifault = 2;
                return value;
        }
        //
        //  Special cases.
        //
        if ( x == 0.0 || x == 1.0 )
        {
                return value;
        }
        //
        //  Change tail if necessary and determine S.
        //
        psq = p + q;
        cx = 1.0 - x;

        if ( p < psq * x )
        {
                xx = cx;
                cx = x;
                pp = q;
                qq = p;
                indx = true;
        }
        else
        {
                xx = x;
                pp = p;
                qq = q;
                indx = false;
        }

        term = 1.0;
        ai = 1.0;
        value = 1.0;
        ns = ( int ) ( qq + cx * psq );
        //
        //  Use the Soper reduction formula.
        //
        rx = xx / cx;
        temp = qq - ai;
        if ( ns == 0 )
        {
                rx = xx;
        }

        for ( ; ; )
        {
                term = term * temp * rx / ( pp + ai );
                value = value + term;;
                temp = fabs ( term );

                if ( temp <= acu && temp <= acu * value )
                {
                        value = value * exp ( pp * log ( xx )
                                              + ( qq - 1.0 ) * log ( cx ) - beta ) / pp;

                        if ( indx )
                        {
                                value = 1.0 - value;
                        }
                        break;
                }

                ai = ai + 1.0;
                ns = ns - 1;

                if ( 0 <= ns )
                {
                        temp = qq - ai;
                        if ( ns == 0 )
                        {
                                rx = xx;
                        }
                }
                else
                {
                        temp = psq;
                        psq = psq + 1.0;
                }
        }

        return value;
}
//****************************************************************************80

number tnc ( number t, number df, number delta, int * ifault )

//****************************************************************************80
//
//  Purpose:
//
//    TNC computes the tail of the noncentral T distribution.
//
//  Discussion:
//
//    This routine computes the cumulative probability at T of the
//    non-central T-distribution with DF degrees of freedom (which may
//    be fractional) and non-centrality parameter DELTA.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 January 2008
//
//  Author:
//
//    Original FORTRAN77 version by Russell Lenth.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Russell Lenth,
//    Algorithm AS 243:
//    Cumulative Distribution Function of the Non-Central T Distribution,
//    Applied Statistics,
//    Volume 38, Number 1, 1989, pages 185-189.
//
//    William Guenther,
//    Evaluation of probabilities for the noncentral distributions and
//    difference of two T-variables with a desk calculator,
//    Journal of Statistical Computation and Simulation,
//    Volume 6, Number 3-4, 1978, pages 199-206.
//
//  Parameters:
//
//    Input, number T, the point whose cumulative probability
//    is desired.
//
//    Input, number DF, the number of degrees of freedom.
//
//    Input, number DELTA, the noncentrality parameter.
//
//    Output, int *IFAULT, error flag.
//    0, no error.
//    nonzero, an error occcurred.
//
//    Output, number TNC, the tail of the noncentral
//    T distribution.
//
{
        number a;
        number albeta;
        number alnrpi = 0.57236494292470008707;
        number b;
        number del;
        number en;
        number errbd;
        number errmax = 1.0E-10;
        number geven;
        number godd;
        //number half;
        int itrmax = 100;
        number lambda;
        bool negdel;
        //number one;
        number p;
        number q;
        number r2pi = 0.79788456080286535588;
        number rxb;
        number s;
        number tt;
        //number two;
        number value;;
        number x;
        number xeven;
        number xodd;
        //number zero;

        value = 0.0;

        if ( df <= 0.0 )
        {
                *ifault = 2;
                return value;
        }

        *ifault = 0;

        tt = t;
        del = delta;
        negdel = false;

        if ( t < 0.0 )
        {
                negdel = true;
                tt = - tt;
                del = - del;
        }
        //
        //  Initialize twin series.
        //
        en = 1.0;
        x = t * t / ( t * t + df );

        if ( x <= 0.0 )
        {
                *ifault = 0;
                value = value + alnorm ( del, true );

                if ( negdel )
                {
                        value = 1.0 - value;
                }
                return value;
        }

        lambda = del * del;
        p = 0.5 * exp ( - 0.5 * lambda );
        q = r2pi * p * del;
        s = 0.5 - p;
        a = 0.5;
        b = 0.5 * df;
        rxb = pow ( 1.0 - x, b );
        albeta = alnrpi + lgamma ( b ) - lgamma ( a + b );
        xodd = betain ( x, a, b, albeta, ifault );
        godd = 2.0 * rxb * exp ( a * log ( x ) - albeta );
        xeven = 1.0 - rxb;
        geven = b * x * rxb;
        value = p * xodd + q * xeven;
        //
        //  Repeat until convergence.
        //
        for ( ; ; )
        {
                a = a + 1.0;
                xodd = xodd - godd;
                xeven = xeven - geven;
                godd = godd * x * ( a + b - 1.0 ) / a;
                geven = geven * x * ( a + b - 0.5 ) / ( a + 0.5 );
                p = p * lambda / ( 2.0 * en );
                q = q * lambda / ( 2.0 * en + 1.0 );
                s = s - p;
                en = en + 1.0;
                value = value + p * xodd + q * xeven;
                errbd = 2.0 * s * ( xodd - godd );

                if ( errbd <= errmax )
                {
                        *ifault = 0;
                        break;
                }

                if ( itrmax < en )
                {
                        *ifault = 1;
                        break;
                }
        }

        value = value + alnorm ( del, true );

        if ( negdel )
        {
                value = 1.0 - value;
        }

        return value;
}

}
