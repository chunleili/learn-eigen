//SparseMatrix1(有Qt版本)

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>
 
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
 
void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n);
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename);
 
int main(int argc, char** argv)
{
  argc = 2;
  argv[1] = "SparseMatrix1.jpg";
  
  int n = 300;  // size of the image
  int m = n*n;  // number of unknowns (=number of pixels)
 
  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  buildProblem(coefficients, b, n);
 
  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());
 
  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
  Eigen::VectorXd x = chol.solve(b);         // use the factorization to solve for the given right hand side
  
  std::ofstream myfile;
  myfile.open ("SparseMatrix1.txt");
  myfile << x;
  myfile.close();

  // Export the result to a file:
  saveAsBitmap(x, n, argv[1]);
 
  return 0;
}
 