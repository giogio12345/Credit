#include "Credit.hpp"

// Spread Kth vs Kth, rho, dim, h and QMC convergence

int main()
{
        using namespace std;

        unsigned dim=5;
        unsigned Kth=1;
        number T=5;
        number npf=1.e6/dim;
        number daycount=0.25;
        number rho=.3;
        number lambda=0.01;
        number r=0.05;
        number R=0.4;
        unsigned nsim=20000000;
        
        {
                cout<<"Price vs Kth (5 names).\n";
                vector<number> Kth_vec= {1,2,3,4,5};
                vector<number> price_g(dim);
                vector<number> price_t(dim);

                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        Kth_to_Default<Gaussian> x(GPU, MC, dim, rho, lambda,
                                                   Kth_vec[i], R, npf, r, T, daycount, nsim, 2);
                        price_g[i]=x.run();
                }

                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        Kth_to_Default<t> x(GPU, MC, dim, rho, lambda,
                                            Kth_vec[i], R, npf, r, T, daycount, nsim, 2);
                        price_t[i]=x.run();
                }
                cout<<"Kth\t";
                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        cout<<Kth_vec[i]<<"\t";
                }
                cout<<"\nG:\t";
                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        cout<<price_g[i]<<"\t";
                }
                cout<<"\nt:\t";
                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        cout<<price_t[i]<<"\t";
                }

                cout<<"\nPrice vs Kth (10 names).\n";
                dim=10;
                nsim/=2;
                Kth_vec= {1,2,3,4,5,6,7,8,9,10};
                price_g.resize(dim);
                price_t.resize(dim);

                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        Kth_to_Default<Gaussian> x(GPU, MC, dim, rho, lambda,
                                                   Kth_vec[i], R, npf, r, T, daycount, nsim, 2);
                        price_g[i]=x.run();
                }

                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        Kth_to_Default<t> x(GPU, MC, dim, rho, lambda,
                                            Kth_vec[i], R, npf, r, T, daycount, nsim, 2);
                        price_t[i]=x.run();
                }
                cout<<"Kth\t";
                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        cout<<Kth_vec[i]<<"\t";
                }
                cout<<"\nG:\t";
                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        cout<<price_g[i]<<"\t";
                }
                cout<<"\nt:\t";
                for (unsigned i=0; i<Kth_vec.size(); ++i)
                {
                        cout<<price_t[i]<<"\t";
                }
        }

        {
                cout<<"\nPrice vs t dof (5 names).\n";
                vector<number> dof_vec= {1,2,3,4,5,6,7,8,9,10};
                unsigned size=dof_vec.size();
                number price_g=0.;
                vector<number> price_t(size);
                nsim=20000000;

                {
                        Kth_to_Default<Gaussian> x(GPU, MC, dim, rho, lambda,
                                                   Kth, R, npf, r, T, daycount, nsim, 2);
                        price_g=x.run();
                }

                for (unsigned i=0; i<size; ++i)
                {
                        Kth_to_Default<t> x(GPU, MC, dim, rho, lambda,
                                            Kth, R, npf, r, T, daycount, nsim, 2, dof_vec[i]);
                        price_t[i]=x.run();
                }
                cout<<"DOF\t";
                for (unsigned i=0; i<size; ++i)
                {
                        cout<<dof_vec[i]<<"\t";
                }
                cout<<"\nG:\t"<<price_g<<"\n";
                cout<<"\nt:\t";
                for (unsigned i=0; i<size; ++i)
                {
                        cout<<price_t[i]<<"\t";
                }
        }

        {
                cout<<"\nPrice vs # names.\n";
                vector<number> names_vec= {5,10,20,50};
                nsim=4000000;
                unsigned size=names_vec.size();
                vector<number> price_g(size);
                vector<number> price_t(size);

                for (unsigned i=0; i<size; ++i)
                {
                        Kth_to_Default<Gaussian> x(GPU, MC, names_vec[i], rho, lambda,
                                                   Kth, R, npf, r, T, daycount, nsim, 2);
                        price_g[i]=x.run();
                }

                for (unsigned i=0; i<size; ++i)
                {
                        Kth_to_Default<t> x(GPU, MC, names_vec[i], rho, lambda,
                                            Kth, R, npf, r, T, daycount, nsim, 2);
                        price_t[i]=x.run();
                }
                cout<<"# names\t";
                for (unsigned i=0; i<size; ++i)
                {
                        cout<<names_vec[i]<<"\t";
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
                cout<<"\nPrice vs rho.\n";
                vector<number> rho_vec= {0., 0.3, 0.7};
                dim=5;
                nsim=20000000;
                Kth=2;
                unsigned size=rho_vec.size();
                vector<number> price_g(size);
                vector<number> price_t(size);

                for (unsigned i=0; i<size; ++i)
                {
                        Kth_to_Default<Gaussian> x(GPU, MC, dim, rho_vec[i], lambda,
                                                   Kth, R, npf, r, T, daycount, nsim, 2);
                        price_g[i]=x.run();
                }

                for (unsigned i=0; i<size; ++i)
                {
                        Kth_to_Default<t> x(GPU, MC, dim, rho_vec[i], lambda,
                                            Kth, R, npf, r, T, daycount, nsim, 2);
                        price_t[i]=x.run();
                }
                cout<<"rho\t";
                for (unsigned i=0; i<size; ++i)
                {
                        cout<<rho_vec[i]<<"\t";
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
                cout<<"\nPrice vs lambda.\n";
                vector<number> lambda_vec= {0.1, 0.01, 0.001};
                unsigned size=lambda_vec.size();
                vector<number> price_g(size);
                vector<number> price_t(size);

                for (unsigned i=0; i<size; ++i)
                {
                        Kth_to_Default<Gaussian> x(GPU, MC, dim, rho, lambda_vec[i],
                                                   Kth, R, npf, r, T, daycount, nsim, 2);
                        price_g[i]=x.run();
                }

                for (unsigned i=0; i<size; ++i)
                {
                        Kth_to_Default<t> x(GPU, MC, dim, rho, lambda_vec[i],
                                            Kth, R, npf, r, T, daycount, nsim, 2);
                        price_t[i]=x.run();
                }
                cout<<"lambda\t";
                for (unsigned i=0; i<size; ++i)
                {
                        cout<<lambda_vec[i]<<"\t";
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
                Kth=1;
                vector<number> names_vec= {5,10,20,50};
                vector<number> nsim_vec= {250000, 500000, 1000000, 2000000};
                unsigned dim_n=names_vec.size();
                unsigned sim_n=nsim_vec.size();

                vector<vector<number> > price_g(dim_n, vector<number>(sim_n));
                vector<vector<number> > price_t(dim_n, vector<number>(sim_n));

                cout<<"MC\n";
                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                Kth_to_Default<Gaussian> x(GPU, MC, names_vec[i], rho, lambda,
                                                           Kth, R, npf, r, T, daycount, nsim_vec[j]);
                                price_g[i][j]=x.run();
                        }
                }

                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                Kth_to_Default<t> x(GPU, MC, names_vec[i], rho, lambda,
                                                    Kth, R, npf, r, T, daycount, nsim_vec[j]);
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
                                Kth_to_Default<Gaussian> x(GPU, QMC, names_vec[i], rho, lambda,
                                                           Kth, R, npf, r, T, daycount, nsim_vec[j]);
                                price_g[i][j]=x.run();
                        }
                }

                for (unsigned i=0; i<dim_n; ++i)
                {
                        for (unsigned j=0; j<sim_n; ++j)
                        {
                                Kth_to_Default<t> x(GPU, QMC, names_vec[i], rho, lambda,
                                                    Kth, R, npf, r, T, daycount, nsim_vec[j]);
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