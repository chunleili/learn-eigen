// main-fillSparseMatrix.cpp
#include <Eigen/Sparse>
#include <Eigen/Dense> //支持dense matrix
#include <unsupported/Eigen/SparseExtra> //支持从matrix market文件读取
#include <iostream>
using namespace Eigen;

const int N = 4, M = 4;

//方法1：使用coeffRef()方法
SparseMatrix<double> fillMatrix1() {
  SparseMatrix<double> m1(N,M);
    // reserve()方法用于预先分配空间，避免后续的内存分配
  m1.reserve(VectorXi::Constant(M, 4)); // 4: estimated number of non-zero enties per column
  m1.coeffRef(0,0) = 1;
  m1.coeffRef(0,1) = 2.;
  m1.coeffRef(1,1) = 3.;
  m1.coeffRef(2,2) = 4.;
  m1.coeffRef(2,3) = 5.;
  m1.coeffRef(3,2) = 6.;
  m1.coeffRef(3,3) = 7.;
  m1.makeCompressed();
  return m1;
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


int main()
{
    auto mat = fillMatrix3();
    std::cout<<"nonzero elements: "<<mat.nonZeros()<<std::endl;
    
}