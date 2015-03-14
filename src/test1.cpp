#include "Credit.hpp"

// SpeedTest CDO 50 names (4M top)

int main()
{
        using namespace std;

        cout<<"SpeedTest CDO 50 names (4M top).\n";

        unsigned dim = 50;         // number of firms
        number R = .4;            // recovery
        number lambda = 0.01;     // (flat) forward default intensity
        number rho = 0.3;          // (flat) correlation between defaults
        number npf = 10000;       // notional per firm
        number a = 0.0;           // attachment point of tranche
        number d = 0.09;           // detachment point of tranche
        number c = 0.05;          // coupon on fixed leg
        number r = 0.03;          // (flat) risk-free interest rate - cc
        //unsigned nsim = 16000000; // number of paths to simulate
        bool isProtectionBuyer = true;  // viewpoint for pricing

        number T = 5;        // "standard 5-year synthetic CDO swap"
        number daycount = 0.5;  // semi-annual coupons
        /*
         number target_n=75.6;
         number target_t=-238.5;
         */
        vector<unsigned> nsim;

        nsim.push_back(250000);
        nsim.push_back(500000);
        nsim.push_back(1000000);
        nsim.push_back(2000000);
        nsim.push_back(4000000);

        {
                Cdo<Gaussian> x(GPU, MC, dim, rho, lambda, R, npf, a, d, c, r, T,
                                daycount, isProtectionBuyer, 10, 2);
                x.run();
        }

        vector<vector<number> > price_n(5, vector<number>(nsim.size(), 0.));
        vector<vector<number> > price_t(5, vector<number>(nsim.size(), 0.));

        vector<vector<number> > IC_n(5, vector<number>(nsim.size(), 0.));
        vector<vector<number> > IC_t(5, vector<number>(nsim.size(), 0.));

        vector<vector<number> > time_n(5, vector<number>(nsim.size(), 0.));
        vector<vector<number> > time_t(5, vector<number>(nsim.size(), 0.));

        vector<vector<number> > speedup_n(3, vector<number>(nsim.size(), 0.));
        vector<vector<number> > speedup_t(3, vector<number>(nsim.size(), 0.));

        // Gaussian

        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<Gaussian> x(CPU, MC, dim, rho, lambda, R, npf, a, d, c, r, T,
                                daycount, isProtectionBuyer, nsim[i]);
                price_n[0][i]=x.run();
                time_n[0][i]=x.get_time();
                IC_n[0][i]=(x.get_ic().second-x.get_ic().first)/2.;
        }


        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<Gaussian> x(CPU, QMC, dim, rho, lambda, R, npf, a, d, c, r, T,
                                daycount, isProtectionBuyer, nsim[i]);
                price_n[1][i]=x.run();
                time_n[1][i]=x.get_time();
                IC_n[1][i]=(x.get_ic().second-x.get_ic().first)/2.;
        }


        cout<<"Cdo MC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<Gaussian> x(GPU, MC, dim, rho, lambda, R, npf, a, d, c, r, T,
                                daycount, isProtectionBuyer, nsim[i], 1);
                price_n[2][i]=x.run();
                time_n[2][i]=x.get_time();
                IC_n[2][i]=(x.get_ic().second-x.get_ic().first)/2.;
                speedup_n[0][i]=time_n[0][i]/time_n[2][i];
                //cout<<x;
        }

        cout<<"Cdo QMC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<Gaussian> x(GPU, QMC, dim, rho, lambda, R, npf, a, d, c, r, T,
                                daycount, isProtectionBuyer, nsim[i], 1);
                price_n[3][i]=x.run();
                time_n[3][i]=x.get_time();
                IC_n[3][i]=(x.get_ic().second-x.get_ic().first)/2.;
                speedup_n[1][i]=time_n[1][i]/time_n[3][i];
        }

        cout<<"Cdo MC 2 GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<Gaussian> x(GPU, MC, dim, rho, lambda, R, npf, a, d, c, r, T,
                                daycount, isProtectionBuyer, nsim[i], 2);
                price_n[4][i]=x.run();
                time_n[4][i]=x.get_time();
                IC_n[4][i]=(x.get_ic().second-x.get_ic().first)/2.;
                speedup_n[2][i]=time_n[0][i]/time_n[4][i];
                //cout<<x;
        }

        // t
        cout<<"\n\nt\n\nCdo MC CPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<t> x(CPU, MC, dim, rho, lambda, R, npf, a, d, c, r, T,
                         daycount, isProtectionBuyer, nsim[i]);
                price_t[0][i]=x.run();
                time_t[0][i]=x.get_time();
                IC_t[0][i]=(x.get_ic().second-x.get_ic().first)/2.;
        }
        cout<<"Cdo QMC CPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<t> x(CPU, QMC, dim, rho, lambda, R, npf, a, d, c, r, T,
                         daycount, isProtectionBuyer, nsim[i]);
                price_t[1][i]=x.run();
                time_t[1][i]=x.get_time();
                IC_t[1][i]=(x.get_ic().second-x.get_ic().first)/2.;
        }

        cout<<"Cdo MC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<t> x(GPU, MC, dim, rho, lambda, R, npf, a, d, c, r, T,
                         daycount, isProtectionBuyer, nsim[i], 1);
                price_t[2][i]=x.run();
                time_t[2][i]=x.get_time();
                IC_t[2][i]=(x.get_ic().second-x.get_ic().first)/2.;
                speedup_t[0][i]=time_t[0][i]/time_t[2][i];
        }

        cout<<"Cdo QMC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<t> x(GPU, QMC, dim, rho, lambda, R, npf, a, d, c, r, T,
                         daycount, isProtectionBuyer, nsim[i], 1);
                price_t[3][i]=x.run();
                time_t[3][i]=x.get_time();
                IC_t[3][i]=(x.get_ic().second-x.get_ic().first)/2.;
                speedup_t[1][i]=time_t[1][i]/time_t[3][i];
        }

        cout<<"Cdo MC GPU.\n";
        for (unsigned i=0; i<nsim.size(); ++i)
        {
                Cdo<t> x(GPU, MC, dim, rho, lambda, R, npf, a, d, c, r, T,
                         daycount, isProtectionBuyer, nsim[i], 2);
                price_t[4][i]=x.run();
                time_t[4][i]=x.get_time();
                IC_t[4][i]=(x.get_ic().second-x.get_ic().first)/2.;
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

                cout<<"\nIC\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_n[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_n[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_n[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_n[3][j]<<"\t";
                }
                cout<<"\nGPU2\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_n[4][j]<<"\t";
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

                cout<<"\nIC\n#sim\t\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<nsim[j]<<"\t";
                }
                cout<<"\nCPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_t[0][j]<<"\t";
                }
                cout<<"\nCPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_t[1][j]<<"\t";
                }
                cout<<"\nGPU\tMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_t[2][j]<<"\t";
                }
                cout<<"\nGPU\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_n[3][j]<<"\t";
                }
                cout<<"\nGPU2\tQMC\t";
                for (unsigned j=0; j<nsim.size(); ++j)
                {
                        cout<<IC_n[4][j]<<"\t";
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
