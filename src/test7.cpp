#include "Credit.hpp"

int main()
{
        using namespace std;

        unsigned nsim=10000000;
        number T=15;
        number R=0.4;

        vector<number> cva_value_u(4,0.);
        vector<number> cva_value_b(4,0.);

        {
                string me("data/unilateral.txt");

                {
                        number a=0.01;
                        number sigma=0.0077937016306329;
                        number c=0.01033;
                        string zc("data/ZC2015.txt");

                        // unilateral
                        {
                                Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                                 me, "data/generali.txt", nsim, true);
                                x.run();
                                cva_value_u[0]=x.get_CVA();
                        }
                        {
                                Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                                 me, "data/unicredit.txt", nsim, true);
                                x.run();
                                cva_value_u[1]=x.get_CVA();
                        }

                        //bilateral
                        {
                                Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                                 "data/generali.txt", "data/unicredit.txt",
                                                 nsim, true);
                                x.run();
                                cva_value_b[0]=x.get_CVA();
                        }
                        {
                                Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                                 "data/unicredit.txt", "data/generali.txt",
                                                 nsim, true);
                                x.run();
                                cva_value_b[1]=x.get_CVA();
                        }
                }
                {
                        number a=0.1;
                        number sigma=0.01;
                        number c=0.04650;
                        string zc("data/ZC2008.txt");

                        // unilateral
                        {
                                Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                                 me, "data/generali2008.txt", nsim, true);
                                x.run();
                                cva_value_u[2]=x.get_CVA();
                        }
                        {
                                Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                                 me, "data/unicredit2008.txt", nsim, true);
                                x.run();
                                cva_value_u[3]=x.get_CVA();
                        }

                        //bilateral
                        {
                                Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                                 "data/generali2008.txt", "data/unicredit2008.txt",
                                                 nsim, true);
                                x.run();
                                cva_value_b[2]=x.get_CVA();
                        }
                        {
                                Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                                 "data/unicredit2008.txt", "data/generali2008.txt",
                                                 nsim, true);
                                x.run();
                                cva_value_b[3]=x.get_CVA();
                        }
                }
        }

        cout<<"*** Unilateral CVA:\nDate\tGenerali\tUnicredit\n2015\t"
            <<cva_value_u[0]<<"\t"<<cva_value_u[1]<<"\n2008\t"
            <<cva_value_u[2]<<"\t"<<cva_value_u[3]<<"\n\n";

        cout<<"*** Bilateral CVA:\n2015\tGenerali\tUniCredit\nGenerali\t**\t"
            <<cva_value_b[0]<<"\nUniCredit\t"<<cva_value_b[1]<<"\t**\n\n";

        cout<<"2008\tGenerali\tUniCredit\nGenerali\t**\t"
        <<cva_value_b[2]<<"\nUniCredit\t"<<cva_value_b[3]<<"\t**\n\n";

        return 0;
}