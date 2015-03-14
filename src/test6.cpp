#include "Credit.hpp"

int main()
{

        using namespace std;

        number c=0.01033;

        string me("data/unilateral.txt");
        string cp("data/unicredit.txt");

        unsigned nsim=1000;
        number T=15;
        number R=0.4;

        cout<<"CPU:\n***HW\n";
        {
                number a=0.01;
                number sigma=0.0077937016306329;
                Cva<HullWhite> x(CPU, MC, T, c, a, sigma, R, R, "data/ZC2015.txt",
                                 me, cp, nsim, true);
                x.set_printfile(true);
                x.run();
                cout<<x;

        }
        cout<<"***V\n";
        {
                number a=0.097916552508002;
                number b=0.017167276938791;
                number sigma=0.010002693189843;
                Cva<Vasicek> x(CPU, MC, T, c, a, b, sigma, R, R, "data/ZC2015.txt",
                               me, cp, nsim, true);
                x.set_printfile(true);
                x.run();
                cout<<x;

        }

        cout<<"Data 2008\n";

        c=0.04650;

        cout<<"CPU:\n***HW\n";
        {
                number a=0.1;
                number sigma=0.01;
                Cva<HullWhite> x(CPU, MC, T, c, a, sigma, R, R, "data/ZC2008.txt",
                                 me, cp, nsim, true);
                x.set_printfile(true);
                x.run();
                cout<<x;

        }
        cout<<"***V\n";
        {
                number a=0.079699185188255;
                number b=0.052093328975552;
                number sigma=0.01;
                Cva<Vasicek> x(CPU, MC, T, c, a, b, sigma, R, R, "data/ZC2008.txt",
                               me, cp, nsim, true);
                x.set_printfile(true);
                x.run();
                cout<<x;

        }
        cout<<"***CIR\n";
        {
                number a=0.03;
                number b=0.050892782179557;
                number sigma=0.05;
                Cva<CIR> x(CPU, MC, T, c, a, b, sigma, R, R, "data/ZC2008.txt", me, cp, nsim, true);
                x.set_printfile(true);
                x.run();
                cout<<x;

        }

        return 0;
}