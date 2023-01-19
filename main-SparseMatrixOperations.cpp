#include <Eigen/Sparse>
#include <iostream>
using namespace Eigen;
using std::cout;
using std::endl;
const int N = 4, M = 4;

SparseMatrix<double> fillMatrix()
{
  SparseMatrix<double> mat(N, M);
  // reserve()方法用于预先分配空间，避免后续的内存分配
  mat.reserve(VectorXi::Constant(M, 4)); // 4: estimated number of non-zero enties per column
  mat.insert(0, 0) = 1;
  mat.insert(0, 1) = 2.;
  mat.insert(1, 1) = 3.;
  mat.insert(2, 2) = 4.;
  mat.insert(2, 3) = 5.;
  mat.insert(3, 2) = 6.;
  mat.insert(3, 3) = 7.;
  mat.makeCompressed();
  return mat;
}

int main()
{
  auto sm1 = fillMatrix();
  auto sm2 = sm1;
  // /* -------------------------------------------------------------------------- */
  // /*                       sparse matrix basic operations                       */
  // /* -------------------------------------------------------------------------- */
  cout<<"sm1:\n"<<sm1<<endl;
  cout<<"sm1.real():\n"<<sm1.real()<<endl;
  // cout<<"sm1.imag():\n"<<sm1.imag()<<endl;
  cout<<"-sm1:\n"<<-sm1<<endl;
  cout<<"0.5*sm1:\n"<<0.5*sm1<<endl;

  cout<<"sm1+sm2:\n"<<sm1+sm2<<endl;
  cout<<"sm1-sm2:\n"<<sm1-sm2<<endl;
  cout<<"sm1.cwiseProduct(sm2):\n"<<sm1.cwiseProduct(sm2)<<endl;

  auto sm3 = sm1;
  auto sm4 = sm1 + sm2 + sm3;
  cout<<"sm4:\n"<<sm4<<endl;

  SparseMatrix<double> A, B;
  B = SparseMatrix<double>(A.transpose()) + A;

  // /* -------------------------------------------------------------------------- */
  // /*                         product with dense matrices                        */
  // /* -------------------------------------------------------------------------- */
  MatrixXd dm1, dm2;
  dm1.setRandom(N, M);
  dm2.setRandom(N, M);

  cout<<"dm1:\n"<<dm1<<endl;
  cout<<"dm2:\n"<<dm2<<endl;

  dm2 = sm1 * dm1;
  cout<<"sm1*dm1:\n"<<dm2<<endl;
  sm2 = sm1.cwiseProduct(dm1);
  cout<<"sm1.cwiseProduct(dm1):\n"<<sm2<<endl;
  dm2 = sm1 + dm1;
  cout<<"sm1+dm1:\n"<<dm2<<endl;
  dm2 = dm1 - sm1;
  cout<<"dm1-sm1:\n"<<dm2<<endl;

  //这比dm2=dm1+sm1要快
  dm2 = dm1;
  dm2 += sm1;

  sm1 = sm2.transpose(); //稀疏矩阵支持转置
  cout<<"sm2.transpose():\n"<<sm1<<endl;
  sm1 = sm2.adjoint();//也支持共轭转置
  cout<<"sm2.adjoint():\n"<<sm1<<endl;

  /* -------------------------------------------------------------------------- */
  /*                                    矩阵乘法                                    */
  /* -------------------------------------------------------------------------- */
  VectorXd dv1,dv2;

  dv2 = sm1 * dv1;
  cout<<"sm1*dv1:\n"<<dv2<<endl;

  dm2 = dm1 * sm1.adjoint();
  cout<<"dm1*sm1.adjoint():\n"<<dm2<<endl;

  dm2 = 2. * sm1 * dm1;
  cout<<"2.*sm1*dm1:\n"<<dm2<<endl;

  // 下面对于对称矩阵
  sm1 = sm1 * sm1.transpose();
  cout << "sm1:\n"
       << sm1 << endl;
  // if only the upper part of sm1 is stored
  cout << "sm1.selfadjointView<Upper>() * dm1:\n"
       << sm1.selfadjointView<Upper>() * dm1 << endl;
  // if only the lower part of sm1 is stored
  cout << "sm1.selfadjointView<Lower>() * dm1:\n"
       << sm1.selfadjointView<Lower>() * dm1 << endl;

  cout << "sm1 * sm2:\n"
       << sm1 * sm2 << endl;
  cout << "4 * sm1.adjoint() * sm2:\n"
       << 4 * sm1.adjoint() * sm2 << endl;

  // 清除稀疏矩阵中的零元素
  cout << "(sm1 * sm2).pruned()\n"
       << (sm1 * sm2).pruned() << endl; // removes numerical zeros
  cout << "(sm1 * sm2).pruned(1e-4)\n"
       << (sm1 * sm2).pruned(1e-4) << endl; // removes elements much smaller than ref
  cout << "(sm1 * sm2).pruned(10,1e-4)\n"
       << (sm1 * sm2).pruned(10, 1e-4) << endl; // removes elements smaller than ref*epsilon

  PermutationMatrix<Dynamic, Dynamic> P;
  P.setIdentity(N);
  sm2 = P * sm1;
  sm2 = sm1 * P.inverse();
  sm2 = sm1.transpose() * P;
    

  /* -------------------------------------------------------------------------- */
  /*                                     块操作                                    */
  /* -------------------------------------------------------------------------- */
  // SparseMatrix<double,ColMajor> sm1;
  // sm1.col(j) = ...;
  // sm1.leftCols(ncols) = ...;
  // sm1.middleCols(j,ncols) = ...;
  // sm1.rightCols(ncols) = ...;
  
  // SparseMatrix<double,RowMajor> sm2;
  // sm2.row(i) = ...;
  // sm2.topRows(nrows) = ...;
  // sm2.middleRows(i,nrows) = ...;
  // sm2.bottomRows(nrows) = ...;

  /* -------------------------------------------------------------------------- */
  /*                                    三角视图                                    */
  /* -------------------------------------------------------------------------- */
  cout<<" sm1.triangularView<Lower>() * dm1"<< sm1.triangularView<Lower>() * dm1;
  cout << "sm1.transpose().triangularView<Upper>() * dv1:\n"
       << sm1.transpose().triangularView<Upper>() * dv1 << endl;
}