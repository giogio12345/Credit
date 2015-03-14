#include "tStudent.cuh"

//****************************************************************************80

__device__ number betain ( number x, number p, number q, number beta/*, int * ifault */)

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
        //*ifault = 0;
        //
        //  Check the input arguments.
        //
        if ( p <= 0.0 || q <= 0.0 )
        {
                //*ifault = 1;
                return value;
        }

        if ( x < 0.0 || 1.0 < x )
        {
                //*ifault = 2;
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
                        value = value * exp ( pp * logf ( xx )
                                              + ( qq - 1.0 ) * logf ( cx ) - beta ) / pp;

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

__device__ number tnc ( number t, number df, number delta/*, int * ifault */)

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
                //*ifault = 2;
                return value;
        }

        //*ifault = 0;

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
                //*ifault = 0;
                value = value + 1-normcdf(del);//alnorm ( del, true );

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
        rxb = pow ( static_cast<number>(1.0) - x, b );
        albeta = alnrpi + lgamma ( b ) - lgamma ( a + b );
        xodd = betain ( x, a, b, albeta/*, ifault */);
        godd = 2.0 * rxb * exp ( a * logf ( x ) - albeta );
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
                        //*ifault = 0;
                        break;
                }

                if ( itrmax < en )
                {
                        //*ifault = 1;
                        break;
                }
        }

        value = value + 1-normcdf(del);//alnorm ( del, true );

        if ( negdel )
        {
                value = 1.0 - value;
        }

        return value;
}