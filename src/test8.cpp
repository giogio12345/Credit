#include "Credit.hpp"

int main()
{
        using namespace std;

        string me("data/unilateral.txt");
        vector<number> T= {5, 10, 15, 30};
        number R=0.4;

        vector<number> EPE_results(8);
        vector<number> CVA_results(8);

        {
                number c=0.04650;
                number a=0.1;
                number sigma=0.01;
                string zc("data/ZC2008.txt");
                unsigned nsim=6000000;
                for (unsigned i=0; i<T.size(); ++i)
                {
                        Cva<HullWhite> x(GPU, MC, T[i], c, a, sigma, R, R, zc,
                                         me, "data/unicredit2008.txt", nsim, true);
                        x.run();
                        EPE_results[i]=x.get_EPE();
                        CVA_results[i]=x.get_CVA();
                }

        }
        {
                number c=0.01033;
                number a=0.01;
                number sigma=0.0077937016306329;
                string zc("data/ZC2015.txt");
                unsigned nsim=6000000;
                for (unsigned i=0; i<T.size(); ++i)
                {
                        Cva<HullWhite> x(GPU, MC, T[i], c, a, sigma, R, R, zc,
                                         me, "data/unicredit.txt", nsim, true);
                        x.run();
                        EPE_results[i+4]=x.get_EPE();
                        CVA_results[i+4]=x.get_CVA();
                }

        }

        cout<<"*** EPE vs T\n\t"<<T[0]<<"\t"<<T[1]<<"\t"<<T[2]<<"\t"<<T[3]<<"\n2008\t"
            <<EPE_results[0]<<"\t"<<EPE_results[1]<<"\t"<<EPE_results[2]<<"\t"<<EPE_results[3]<<"\n2015\t"
            <<EPE_results[4]<<"\t"<<EPE_results[5]<<"\t"<<EPE_results[6]<<"\t"<<EPE_results[7]<<"\n";

        cout<<"*** CVA vs T\n\t"<<T[0]<<"\t"<<T[1]<<"\t"<<T[2]<<"\t"<<T[3]<<"\n2008\t"
            <<CVA_results[0]<<"\t"<<CVA_results[1]<<"\t"<<CVA_results[2]<<"\t"<<CVA_results[3]<<"\n2015\t"
            <<CVA_results[4]<<"\t"<<CVA_results[5]<<"\t"<<CVA_results[6]<<"\t"<<CVA_results[7]<<"\n";


        return 0;
}