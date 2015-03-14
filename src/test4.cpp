#include "Credit.hpp"

// Spread CDO vs tranches and QMC convergence

int main()
{
        using namespace std;

        unsigned dim = 500;
        number R = .4;
        number lambda = 0.01;
        number rho = 0.3;
        number npf = 10000;
        number c = 0.05;
        number r = 0.03;
        unsigned nsim=800000;
        bool isProtectionBuyer = true;

        number T = 5;
        number daycount = 0.5;
        
        {
                cout<<"Price vs Tranches.\n";
                vector<pair<number, number> > tranches=
                {
                        make_pair<number, number>(0.0, 0.09),
                        make_pair<number, number>(0.09, 0.12),
                        make_pair<number, number>(0.12, 0.3)
                };
                unsigned size=tranches.size();
                vector<number> price_g(size);
                vector<number> price_t(size);

                for (unsigned i=0; i<size; ++i)
                {
                        Cdo<Gaussian> x(GPU, MC, dim, rho, lambda, R, npf, tranches[i].first,
                                        tranches[i].second, c, r, T, daycount, isProtectionBuyer,
                                        nsim, 2);
                        price_g[i]=x.run();
                }

                for (unsigned i=0; i<size; ++i)
                {
                        Cdo<t> x(GPU, MC, dim, rho, lambda, R, npf, tranches[i].first,
                                 tranches[i].second, c, r, T, daycount, isProtectionBuyer,
                                 nsim, 2);
                        price_t[i]=x.run();
                }
                cout<<"Tranches:\t";
                for (unsigned i=0; i<size; ++i)
                {
                        cout<<"("<<tranches[i].first<<", "<<tranches[i].second<<")\t";
                }
                cout<<"\nG:\t";
                for (unsigned i=0; i<size; ++i)
                {
                        cout<<price_g[i]<<"\t";
                }
                cout<<"\nt:\t";
                for (unsigned i=0; i<size; ++i)
                {
                        cout<<price_t[i]<<"\t";
                }

        }
        
        {
                cout<<"\nMC/QMC Convergence vs Dimension.\n";
                number a=0.;
                number d=0.1;
                vector<number> names_vec= {10,50,100,500,1000};
                vector<number> nsim_vec= {25000, 50000, 100000, 200000};
                unsigned dim_n=names_vec.size();
                unsigned sim_n=nsim_vec.size();

                vector<vector<number> > price_g(dim_n, vector<number>(sim_n));
                vector<vector<number> > price_t(dim_n, vector<number>(sim_n));

                cout<<"MC\n";
                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                Cdo<Gaussian> x(GPU, MC, names_vec[i], rho, lambda, R, npf, a,
                                                d, c, r, T, daycount, isProtectionBuyer,
                                                nsim_vec[j]);
                                price_g[i][j]=x.run();
                        }
                }

                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                Cdo<t> x(GPU, MC, names_vec[i], rho, lambda, R, npf, a,
                                         d, c, r, T, daycount, isProtectionBuyer,
                                         nsim_vec[j]);
                                price_t[i][j]=x.run();
                        }
                }

                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                price_g[i][j]=fabs(price_g[i][j]-price_g[i][sim_n-1])/
                                              (price_g[i][sim_n-1]);
                                price_t[i][j]=fabs(price_t[i][j]-price_t[i][sim_n-1])/
                                              (price_t[i][sim_n-1]);
                        }
                }

                cout<<"G:\ndim/sim\t";
                for (unsigned i=0; i<sim_n; ++i)
                {
                        cout<<nsim_vec[i]<<"\t";
                }
                for (unsigned i=0; i<dim_n; ++i)
                {
                        cout<<"\n"<<names_vec[i]<<"\t";
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                cout<<price_g[i][j]<<"\t";
                        }
                }
                cout<<"\nt:\ndim/sim\t";
                for (unsigned i=0; i<sim_n; ++i)
                {
                        cout<<nsim_vec[i]<<"\t";
                }
                for (unsigned i=0; i<dim_n; ++i)
                {
                        cout<<"\n"<<names_vec[i]<<"\t";
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                cout<<price_t[i][j]<<"\t";
                        }
                }

                cout<<"\nQMC\n";
                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                Cdo<Gaussian> x(GPU, QMC, names_vec[i], rho, lambda, R, npf, a,
                                                d, c, r, T, daycount, isProtectionBuyer,
                                                nsim_vec[j]);
                                price_g[i][j]=x.run();
                        }
                }

                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                Cdo<t> x(GPU, QMC, names_vec[i], rho, lambda, R, npf, a,
                                         d, c, r, T, daycount, isProtectionBuyer,
                                         nsim_vec[j]);
                                price_t[i][j]=x.run();
                        }
                }

                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                price_g[i][j]=fabs(price_g[i][j]-price_g[i][sim_n-1])/
                                              (price_g[i][sim_n-1]);
                                price_t[i][j]=fabs(price_t[i][j]-price_t[i][sim_n-1])/
                                              (price_t[i][sim_n-1]);
                        }
                }

                cout<<"G:\ndim/sim\t";
                for (unsigned i=0; i<sim_n; ++i)
                {
                        cout<<nsim_vec[i]<<"\t";
                }
                for (unsigned i=0; i<dim_n; ++i)
                {
                        cout<<"\n"<<names_vec[i]<<"\t";
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                cout<<price_g[i][j]<<"\t";
                        }
                }
                cout<<"\nt:\ndim/sim\t";
                for (unsigned i=0; i<sim_n; ++i)
                {
                        cout<<nsim_vec[i]<<"\t";
                }
                for (unsigned i=0; i<dim_n; ++i)
                {
                        cout<<"\n"<<names_vec[i]<<"\t";
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                cout<<price_t[i][j]<<"\t";
                        }
                }
                cout<<"\n";
        }

        return 0;
}