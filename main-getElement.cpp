// main-fillSparseMatrix.cpp
#include <Eigen/Sparse>
#include <Eigen/Dense>                   //支持dense matrix
#include <unsupported/Eigen/SparseExtra> //支持从matrix market文件读取
#include <iostream>
using namespace Eigen;

const int N = 4, M = 4;

// 方法1：使用coeffRef()方法
SparseMatrix<double> fillMatrix1()
{
    SparseMatrix<double> mat(N, M);
    // reserve()方法用于预先分配空间，避免后续的内存分配
    mat.reserve(VectorXi::Constant(M, 4)); // 4: estimated number of non-zero enties per column
    mat.coeffRef(0, 0) = 1;
    mat.coeffRef(0, 1) = 2.;
    mat.coeffRef(1, 1) = 3.;
    mat.coeffRef(2, 2) = 4.;
    mat.coeffRef(2, 3) = 5.;
    mat.coeffRef(3, 2) = 6.;
    mat.coeffRef(3, 3) = 7.;
    mat.makeCompressed();
    return mat;
}

// 方法2：从dense matrix转换而来
SparseMatrix<double> fillMatrix2()
{
    MatrixXd mat_dense = MatrixXd::Random(N, M);
    Eigen::SparseMatrix<double> mat = mat_dense.sparseView(0.5, 1);
    return mat;
}

// 方法3：从matrix market文件读取
SparseMatrix<double> fillMatrix3()
{
    SparseMatrix<double> mat;
    Eigen::loadMarket(mat, "e05r0500.mtx");
    return mat;
}

// 方法4：使用setFromTriplets()方法(官方推荐)
SparseMatrix<double> fillMatrix4()
{
    typedef Eigen::Triplet<double> T;
    std::vector<T> coefficients;

    coefficients.push_back(T(0, 0, 2)); // row, col, value
    coefficients.push_back(T(0, 1, 6));
    coefficients.push_back(T(1, 0, 6));
    coefficients.push_back(T(1, 1, 1));
    coefficients.push_back(T(2, 2, 5));

    Eigen::SparseMatrix<double> mat(N, M);
    mat.setFromTriplets(coefficients.begin(), coefficients.end());
    return mat;
}

// 方法5：使用insert()方法, 效率比方法4稍低但内存消耗小
SparseMatrix<double> fillMatrix5()
{
    SparseMatrix<double> mat(N, M); // default is column major
    mat.reserve(VectorXi::Constant(M, 6));
    // alternative: mat.coeffRef(i,j) += v_ij; 但这样需要二叉搜索, 效率低
    mat.insert(0, 0) = 2;
    mat.insert(0, 1) = 6;
    mat.insert(1, 0) = 6;
    mat.insert(1, 1) = 1;
    mat.insert(2, 2) = 5;
    mat.makeCompressed(); // optional
    return mat;
}

int main()
{
    auto mat = fillMatrix5();
    std::cout << "mat[0,0]: " << mat.coeff(0, 0) << std::endl;
    mat.coeffRef(0,0) = 3;
    // type of mat.coeff(0,0) is const double&
    std::cout<<typeid(mat.coeff(0,0)).name()<<std::endl;
    std::cout << "mat[0,0]: " << mat.coeffRef(0, 0) << std::endl;
    std::cout << "nonzero elements: " << mat.nonZeros() << std::endl;
    std::cout << mat << std::endl;
}