// Cholesky分解
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
  
int main()
{
  std::vector<Eigen::Triplet<double>> coefficients;

  coefficients.push_back(Eigen::Triplet<double>(0,0,2));
  coefficients.push_back(Eigen::Triplet<double>(0,1,6));
  coefficients.push_back(Eigen::Triplet<double>(1,0,6));
  coefficients.push_back(Eigen::Triplet<double>(1,1,1));
  coefficients.push_back(Eigen::Triplet<double>(2,2,5));

  Eigen::SparseMatrix<double> A(3,3);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  std::cout << A << std::endl;

  // perform the Cholesky factorization of A:
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(A);

  Eigen::VectorXd b(3);
  b << 1, 2, 3;

  // use the factorization to solve for the given right hand side:
  Eigen::VectorXd x = chol.solve(b);

  std::cout << x << std::endl;

  return 0;
}