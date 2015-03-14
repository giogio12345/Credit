#include "Credit.hpp"

// SpeedTest Kth 5 names (40M top)

int main()
{
        using namespace std;

        cout<<"SpeedTest Kth 5 names (40M top).\n";

        unsigned dim=5;          // Problem dimension
        unsigned Kth=1;           // K-th to default
        number T=5;               // Time To Maturity
        number npf=1.e6/dim;            // Notional amount
        number daycount=0.25;         // 1 payment for year
        number rho=.3;            // Correlation index between each pair
        number lambda=0.01;       // Default intensity
        number r=0.05;            // Risk free interest rate
        number R=0.4;             // Recovery rate

        vector<unsigned> nsim;
        nsim.push_back(2500000);
        nsim.push_back(5000000);
        nsim.push_back(10000000);
        nsim.push_back(20000000);

        {
                Kth_to_Default<Gaussian> x(GPU, MC, dim, rho, lambda,
                                           Kth, R, npf, r, T, daycount, 10, 2);
                x.run();
        }

        vector<vector<number> > price_n(5, vector<number>(nsim.size(), 0.));
        vector<vector<number> > price_t(5, vector<number>(nsim.size(), 0.));

        vector<vector<number> > time_n(5, vector<number>(nsim.size(), 0.));
        vector<vector<number> > time_t(5, vector<number>(nsim.size(), 0.));

        vector<vector<number> > speedup_n(3, vector<number>(nsim.size(), 0.));
        vector<vector<number> > speedup_t(3, vector<number>(nsim.size(), 0.));

        // Gaussian

        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<Gaussian> x(CPU, MC, dim, rho, lambda,
                                           Kth, R, npf, r, T, daycount, nsim[i]);
                price_n[0][i]=x.run();
                time_n[0][i]=x.get_time();
        }


        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<Gaussian> x(CPU, QMC, dim, rho, lambda,
                                           Kth, R, npf, r, T, daycount, nsim[i]);
                price_n[1][i]=x.run();
                time_n[1][i]=x.get_time();
        }


        cout<<"Kth_to_Default MC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<Gaussian> x(GPU, MC, dim, rho, lambda,
                                           Kth, R, npf, r, T, daycount, nsim[i], 1);
                price_n[2][i]=x.run();
                time_n[2][i]=x.get_time();
                speedup_n[0][i]=time_n[0][i]/time_n[2][i];
        }

        cout<<"Kth_to_Default QMC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<Gaussian> x(GPU, QMC, dim, rho, lambda,
                                           Kth, R, npf, r, T, daycount, nsim[i], 1);
                price_n[3][i]=x.run();
                time_n[3][i]=x.get_time();
                speedup_n[1][i]=time_n[1][i]/time_n[3][i];
        }

        cout<<"Kth_to_Default MC 2 GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<Gaussian> x(GPU, MC, dim, rho, lambda,
                                           Kth, R, npf, r, T, daycount, nsim[i], 2);
                price_n[4][i]=x.run();
                time_n[4][i]=x.get_time();
                speedup_n[2][i]=time_n[0][i]/time_n[4][i];
        }

        // t
        cout<<"\n\nt\n\nKth_to_Default MC CPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<t> x(CPU, MC, dim, rho, lambda,
                                    Kth, R, npf, r, T, daycount, nsim[i]);
                price_t[0][i]=x.run();
                time_t[0][i]=x.get_time();
        }
        cout<<"Kth_to_Default QMC CPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<t> x(CPU, QMC, dim, rho, lambda,
                                    Kth, R, npf, r, T, daycount, nsim[i]);
                price_t[1][i]=x.run();
                time_t[1][i]=x.get_time();
        }

        cout<<"Kth_to_Default MC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<t> x(GPU, MC, dim, rho, lambda,
                                    Kth, R, npf, r, T, daycount, nsim[i], 1);
                price_t[2][i]=x.run();
                time_t[2][i]=x.get_time();
                speedup_t[0][i]=time_t[0][i]/time_t[2][i];
        }

        cout<<"Kth_to_Default QMC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<t> x(GPU, QMC, dim, rho, lambda,
                                    Kth, R, npf, r, T, daycount, nsim[i], 1);
                price_t[3][i]=x.run();
                time_t[3][i]=x.get_time();
                speedup_t[1][i]=time_t[1][i]/time_t[3][i];
        }

        cout<<"Kth_to_Default MC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Kth_to_Default<t> x(GPU, MC, dim, rho, lambda,
                                    Kth, R, npf, r, T, daycount, nsim[i], 2);
                price_t[4][i]=x.run();
                time_t[4][i]=x.get_time();
                speedup_t[2][i]=time_t[0][i]/time_t[4][i];
        }

        {
                cout<<"*** Gaussian:\n";
                cout<<"Prices\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_n[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_n[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_n[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_n[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_n[4][j]<<"\t";
                }

                cout<<"\nTime\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_n[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_n[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_n[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_n[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_n[4][j]<<"\t";
                }

                cout<<"\nSpeedup\n#sim\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_n[0][j]<<"\t";
                }
                cout<<"\nQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_n[1][j]<<"\t";
                }
                cout<<"\nMC 2GPU\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_n[2][j]<<"\t";
                }

        }

        {
                cout<<"\n\n*** Student's t:\n";
                cout<<"Prices\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_t[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_t[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_t[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_t[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<price_t[4][j]<<"\t";
                }

                cout<<"\nTime\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_t[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_t[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_t[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_t[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<time_t[4][j]<<"\t";
                }

                cout<<"\nSpeedup\n#sim\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_t[0][j]<<"\t";
                }
                cout<<"\nQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_t[1][j]<<"\t";
                }
                cout<<"\nMC 2GPU\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<speedup_t[2][j]<<"\t";
                }
                cout<<"\n\n";

        }

        return 0;
}
