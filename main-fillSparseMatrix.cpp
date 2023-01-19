// main-fillSparseMatrix.cpp
#include <Eigen/Sparse>
#include <Eigen/Dense> //支持dense matrix
#include <unsupported/Eigen/SparseExtra> //支持从matrix market文件读取
#include <iostream>
using namespace Eigen;

const int N = 4, M = 4;

//方法1：使用coeffRef()方法
SparseMatrix<double> fillMatrix1() {
  SparseMatrix<double> mat(N,M);
  // reserve()方法用于预先分配空间，避免后续的内存分配
  mat.reserve(VectorXi::Constant(M, 4)); // 4: estimated number of non-zero enties per column
  mat.coeffRef(0,0) = 1;
  mat.coeffRef(0,1) = 2.;
  mat.coeffRef(1,1) = 3.;
  mat.coeffRef(2,2) = 4.;
  mat.coeffRef(2,3) = 5.;
  mat.coeffRef(3,2) = 6.;
  mat.coeffRef(3,3) = 7.;
  mat.makeCompressed();
  return mat;
}

//方法2：从dense matrix转换而来
SparseMatrix<double> fillMatrix2() {
    MatrixXd mat_dense = MatrixXd::Random(N,M);
    Eigen::SparseMatrix<double> mat = mat_dense.sparseView(0.5, 1);
    return mat;
}

//方法3：从matrix market文件读取
SparseMatrix<double> fillMatrix3() {
    SparseMatrix<double> mat;
    Eigen::loadMarket(mat, "e05r0500.mtx");
    return mat;
}

//方法4：使用setFromTriplets()方法
SparseMatrix<double> fillMatrix4() {
  std::vector<Eigen::Triplet<double>> coefficients;

  coefficients.push_back(Eigen::Triplet<double>(0,0,2));//row, col, value
  coefficients.push_back(Eigen::Triplet<double>(0,1,6));
  coefficients.push_back(Eigen::Triplet<double>(1,0,6));
  coefficients.push_back(Eigen::Triplet<double>(1,1,1));
  coefficients.push_back(Eigen::Triplet<double>(2,2,5));

  Eigen::SparseMatrix<double> mat(N,M);
  mat.setFromTriplets(coefficients.begin(), coefficients.end());
  return mat
}


int main()
{
    auto mat = fillMatrix4();
    std::cout<<"nonzero elements: "<<mat.nonZeros()<<std::endl;
    
}