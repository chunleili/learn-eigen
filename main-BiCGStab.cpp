#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
using namespace Eigen;


int main()
{
    VectorXd x, b;
    SparseMatrix<double> A;

    /* ... fill A and b ... */
    Eigen::loadMarket(A, "e05r0500.mtx");
    Eigen::loadMarketVector(b, "e05r0500_rhs1.mtx");

    std::cout<<"A: "<<A.rows()<<" "<<A.cols() <<std::endl;
    std::cout<<"b: "<<b.rows()<<" "<<b.cols() <<std::endl;

    BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> solver;
    solver.compute(A);
    x = solver.solve(b);
    std::cout << "#iterations:     " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error() << std::endl;
    std::cout<< "solver info: "<< solver.info() <<std::endl;
    std::cout<< "Eigen::Success "<< Eigen::Success<<std::endl;
    
}