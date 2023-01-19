//SparseMatrix1（去掉Qt的部分）

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <fstream>
#include "utils.h"

/* -------------------------------------------------------------------------- */
/*                            构建稀疏矩阵，从Laplace方程构建                    */
/* -------------------------------------------------------------------------- */
#define _USE_MATH_DEFINES
#include <math.h>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
                       Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
  int n = int(boundary.size());
  int id1 = i+j*n;
 
        if(i==-1 || i==n) b(id) -= w * boundary(j); // constrained coefficient
  else  if(j==-1 || j==n) b(id) -= w * boundary(i); // constrained coefficient
  else  coeffs.push_back(T(id,id1,w));              // unknown coefficient
}
 
void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
  b.setZero();
  Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0,M_PI).sin().pow(2);
  for(int j=0; j<n; ++j)
  {
    for(int i=0; i<n; ++i)
    {
      int id = i+j*n;
      insertCoefficient(id, i-1,j, -1, coefficients, b, boundary);
      insertCoefficient(id, i+1,j, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j-1, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j+1, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j,    4, coefficients, b, boundary);
    }
  }
}


/* -------------------------------------------------------------------------- */
/*                                    求解矩阵                                 */
/* -------------------------------------------------------------------------- */

int main(int argc, char** argv)
{
  int n = 300;  // size of the image
  int m = n*n;  // number of unknowns (=number of pixels)

  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  buildProblem(coefficients, b, n);
 
  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());
 
  Profiler p;
  p.start();
  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
  Eigen::VectorXd x = chol.solve(b);         // use the factorization to solve for the given right hand side
  p.end();
  
  if (chol.info()==Eigen::Success)
    std::cout << "Solve Success" << std::endl;
  else
    std::cout << "Fail" << std::endl;

  std::ofstream myfile;
  myfile.open ("SparseMatrix1.txt");
  myfile << x;
  myfile.close();
 
  return 0;
}