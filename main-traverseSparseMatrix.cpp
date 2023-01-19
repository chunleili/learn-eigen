// main-traverseSparseMatrix.cpp
#include <Eigen/Sparse>
#include <iostream>
using namespace Eigen;

//遍历非零元素（如果要输出所有元素，直接cout<<A即可,但不建议）
void traverseSparseMatrix(const SparseMatrix<double>& mat) {
    for (int k = 0; k < mat.outerSize(); ++k)
        for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
        {
            it.value();
            it.row();   // row index
            it.col();   // col index (here it is equal to k)
            it.index(); // inner index, here it is equal to it.row()

            static int step = 0;
            std::cout << "step = " << step++ << std::endl;
            std::cout << "it.value() = " << it.value() << std::endl;
            // std::cout << "it.row() = " << it.row() << std::endl;
            // std::cout << "it.col() = " << it.col() << std::endl;
            // std::cout << "it.index() = " << it.index() << std::endl;
        }
}


int main()
{
    int n = 10, m = 5;
    MatrixXd mat_dense = MatrixXd::Random(n,m);
    Eigen::SparseMatrix<double> mat = mat_dense.sparseView(0.5, 1);
    std::cout<<"nonzero elements: "<<mat.nonZeros()<<std::endl;
    TraverseSparseMatrix(mat);
}