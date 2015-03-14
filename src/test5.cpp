#include "Credit.hpp"

int main()
{
        using namespace std;

        number c=0.04650;
        string zc("data/ZC2008.txt");
        string me("data/unilateral.txt");
        string cp("data/unicredit.txt");
        number T=15;
        number R=0.4;

        vector<number> nsim= {1250000, 2500000, 5000000, 10000000};
        vector<number> nsim_CIR= {250000, 500000, 1000000, 2000000};

        vector<vector<number> > EPE_HW  (5, vector<number>(nsim.size()));
        vector<vector<number> > EPE_V   (5, vector<number>(nsim.size()));
        vector<vector<number> > EPE_CIR (3, vector<number>(nsim.size()));

        vector<vector<number> > CVA_HW  (5, vector<number>(nsim.size()));
        vector<vector<number> > CVA_V   (5, vector<number>(nsim.size()));
        vector<vector<number> > CVA_CIR (3, vector<number>(nsim.size()));

        vector<vector<number> > time_HW  (5, vector<number>(nsim.size()));
        vector<vector<number> > time_V   (5, vector<number>(nsim.size()));
        vector<vector<number> > time_CIR (3, vector<number>(nsim.size()));

        vector<vector<number> > speedup_HW  (3, vector<number>(nsim.size()));
        vector<vector<number> > speedup_V   (3, vector<number>(nsim.size()));
        vector<vector<number> > speedup_CIR (2, vector<number>(nsim.size()));
        // Hull-White
        {
                number a=0.1;
                number sigma=0.01;

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<HullWhite> x(CPU, MC, T, c, a, sigma, R, R, zc,
                                         me, cp, nsim[i], true);
                        x.run();
                        EPE_HW[0][i]=x.get_EPE();
                        CVA_HW[0][i]=x.get_CVA();
                        time_HW[0][i]=x.get_time();
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<HullWhite> x(CPU, QMC, T, c, a, sigma, R, R, zc,
                                         me, cp, nsim[i], true);
                        x.run();
                        EPE_HW[1][i]=x.get_EPE();
                        CVA_HW[1][i]=x.get_CVA();
                        time_HW[1][i]=x.get_time();
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                         me, cp, nsim[i], true);
                        x.run();
                        EPE_HW[2][i]=x.get_EPE();
                        CVA_HW[2][i]=x.get_CVA();
                        time_HW[2][i]=x.get_time();
                        speedup_HW[0][i]=time_HW[0][i]/time_HW[2][i];
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<HullWhite> x(GPU, QMC, T, c, a, sigma, R, R, zc,
                                         me, cp, nsim[i], true);
                        x.run();
                        EPE_HW[3][i]=x.get_EPE();
                        CVA_HW[3][i]=x.get_CVA();
                        time_HW[3][i]=x.get_time();
                        speedup_HW[1][i]=time_HW[1][i]/time_HW[3][i];
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<HullWhite> x(GPU, MC, T, c, a, sigma, R, R, zc,
                                         me, cp, nsim[i], true, 2);
                        x.run();
                        EPE_HW[4][i]=x.get_EPE();
                        CVA_HW[4][i]=x.get_CVA();
                        time_HW[4][i]=x.get_time();
                        speedup_HW[2][i]=time_HW[0][i]/time_HW[4][i];
                }
        }
        // Vasicek
        {
                number a=0.079699185188255;
                number b=0.052093328975552;
                number sigma=0.01;

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<Vasicek> x(CPU, MC, T, c, a, b, sigma, R, R, zc,
                                       me, cp, nsim[i], true);
                        x.run();
                        EPE_V[0][i]=x.get_EPE();
                        CVA_V[0][i]=x.get_CVA();
                        time_V[0][i]=x.get_time();
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<Vasicek> x(CPU, QMC, T, c, a, b, sigma, R, R, zc,
                                       me, cp, nsim[i], true);
                        x.run();
                        EPE_V[1][i]=x.get_EPE();
                        CVA_V[1][i]=x.get_CVA();
                        time_V[1][i]=x.get_time();
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<Vasicek> x(GPU, MC, T, c, a, b, sigma, R, R, zc,
                                       me, cp, nsim[i], true);
                        x.run();
                        EPE_V[2][i]=x.get_EPE();
                        CVA_V[2][i]=x.get_CVA();
                        time_V[2][i]=x.get_time();
                        speedup_V[0][i]=time_V[0][i]/time_V[2][i];
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<Vasicek> x(GPU, QMC, T, c, a, b, sigma, R, R, zc,
                                       me, cp, nsim[i], true);
                        x.run();
                        EPE_V[3][i]=x.get_EPE();
                        CVA_V[3][i]=x.get_CVA();
                        time_V[3][i]=x.get_time();
                        speedup_V[1][i]=time_V[1][i]/time_V[3][i];
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<Vasicek> x(GPU, MC, T, c, a, b, sigma, R, R, zc,
                                       me, cp, nsim[i], true, 2);
                        x.run();
                        EPE_V[4][i]=x.get_EPE();
                        CVA_V[4][i]=x.get_CVA();
                        time_V[4][i]=x.get_time();
                        speedup_V[2][i]=time_V[0][i]/time_V[4][i];
                }
        }
        // CIR
        {
                number a=0.03;
                number b=0.050892782179557;
                number sigma=0.05;

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<CIR> x(CPU, MC, T, c, a, b, sigma, R, R, zc,
                                   me, cp, nsim_CIR[i], true);
                        x.run();
                        EPE_CIR[0][i]=x.get_EPE();
                        CVA_CIR[0][i]=x.get_CVA();
                        time_CIR[0][i]=x.get_time();
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<CIR> x(GPU, MC, T, c, a, b, sigma, R, R, zc,
                                   me, cp, nsim_CIR[i], true);
                        x.run();
                        EPE_CIR[1][i]=x.get_EPE();
                        CVA_CIR[1][i]=x.get_CVA();
                        time_CIR[1][i]=x.get_time();
                        speedup_CIR[0][i]=time_CIR[0][i]/time_CIR[1][i];
                }

                for (unsigned i=0; i<nsim.size(); ++i)
                {
                        Cva<CIR> x(GPU, MC, T, c, a, b, sigma, R, R, zc,
                                   me, cp, nsim_CIR[i], true, 2);
                        x.run();
                        EPE_CIR[2][i]=x.get_EPE();
                        CVA_CIR[2][i]=x.get_CVA();
                        time_CIR[2][i]=x.get_time();
                        speedup_CIR[1][i]=time_CIR[0][i]/time_CIR[2][i];
                }
        }

        // Hull-White
        {
                cout<<"*** Hull-White:\n";
                cout<<"EPE\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_HW[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_HW[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_HW[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_HW[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_HW[4][j]<<"\t";
                }

                cout<<"\nCVA\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_HW[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_HW[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_HW[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_HW[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_HW[4][j]<<"\t";
                }

                cout<<"\nTime\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_HW[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_HW[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_HW[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_HW[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_HW[4][j]<<"\t";
                }

                cout<<"\nSpeedup\n#sim\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_HW[0][j]<<"\t";
                }
                cout<<"\nQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_HW[1][j]<<"\t";
                }
                cout<<"\nMC 2GPU\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_HW[2][j]<<"\t";
                }

        }
        // Vasicek
        {
                cout<<"\n*** Vasicek:\n";
                cout<<"EPE\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_V[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_V[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_V[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_V[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<EPE_V[4][j]<<"\t";
                }

                cout<<"\nCVA\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_V[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_V[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_V[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_V[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<CVA_V[4][j]<<"\t";
                }

                cout<<"\nTime\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_V[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_V[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_V[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_V[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_V[4][j]<<"\t";
                }

                cout<<"\nSpeedup\n#sim\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_V[0][j]<<"\t";
                }
                cout<<"\nQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_V[1][j]<<"\t";
                }
                cout<<"\nMC 2GPU\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_V[2][j]<<"\t";
                }

        }
        // CIR
        {
                cout<<"\n*** CIR:\n";
                cout<<"EPE\n#sim\t\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<nsim_CIR[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<EPE_CIR[0][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<EPE_CIR[1][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<EPE_CIR[2][j]<<"\t";
                }

                cout<<"\nCVA\n#sim\t\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<nsim_CIR[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<CVA_CIR[0][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<CVA_CIR[1][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<CVA_CIR[2][j]<<"\t";
                }

                cout<<"\nTime\n#sim\t\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<time_CIR[0][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<time_CIR[1][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<time_CIR[2][j]<<"\t";
                }

                cout<<"\nSpeedup\n#sim\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nMC\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<speedup_CIR[0][j]<<"\t";
                }
                cout<<"\nMC 2GPU\t";
                for (unsigned j=0; j<nsim_CIR.size(); ++j)
                {
                        cout<<speedup_CIR[1][j]<<"\t";
                }
                cout<<"\n";
        }

        return 0;
}