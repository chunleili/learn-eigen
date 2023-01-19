// matrix-free solver
#include <iostream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

class MatrixReplacement; // 提前声明一个包装类
using Eigen::SparseMatrix;
const int N = 100;

// 定义一个traits模板类
// 如果不定义，则报错：C2027 使用了未定义类型“Eigen::internal::traits<Derived>”
namespace Eigen
{
  namespace internal
  {
    // MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
    template <>
    struct traits<MatrixReplacement> : public Eigen::internal::traits<Eigen::SparseMatrix<double>>
    {};
  }
}

// 定义一个包装类模板MatrixReplacement，这个类模板是核心。这个类用来包装sparse matrix
//  Example of a matrix-free wrapper from a user type to Eigen's compatible type
//  For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement>
{
public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum
  {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  // MYADD 要更改的地方1：返回行号和列号的函数
  Index rows() const { return mp_mat->rows(); }
  Index cols() const { return mp_mat->cols(); }

  // MYADD 要更改的地方2：定义乘法运算符。
  template <typename Rhs>
  Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const
  {
    return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // Custom API:
  // 默认构造函数，给个0值
  MatrixReplacement() : mp_mat(0) {}

  // 真正把矩阵传进来的函数
  void attachMyMatrix(const SparseMatrix<double> &mat){
    mp_mat = &mat;
  }

  // 返回矩阵的函数
  const SparseMatrix<double> my_matrix() const { return *mp_mat; }

private:
  // 被存储的矩阵数据的指针
  const SparseMatrix<double> *mp_mat;
};

// 真正定义乘法运算符的地方
//  Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen
{
  namespace internal
  {

    // 包装类MatrixReplacement和DenseVector的乘法实现
    template <typename Rhs>
    struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
        : generic_product_impl_base<MatrixReplacement, Rhs, generic_product_impl<MatrixReplacement, Rhs>>
    {
      typedef typename Product<MatrixReplacement, Rhs>::Scalar Scalar;

      // MYADD: 乘以并加上,最重要的要实现的地方
      template <typename Dest>
      static void scaleAndAddTo(Dest &dst, const MatrixReplacement &lhs, const Rhs &rhs, const Scalar &alpha)
      {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        assert(alpha == Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);

        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        // but let's do something fancier (and less efficient):
        dst.noalias() += lhs.my_matrix() * rhs;
        // for (Index i = 0; i < lhs.cols(); ++i)
        // dst += rhs(i) * lhs.my_matrix().col(i);
      }
    };
  }
}

template <typename TSolver=Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner>>
void linSol(MatrixReplacement &A, Eigen::VectorXd &b, Eigen::VectorXd &x)
{
  std::string solverName = typeid(TSolver).name();
  auto pos1 = solverName.find_first_of("::")+2;
  auto pos2 = solverName.find_first_of("<");
  solverName = solverName.substr(pos1, pos2 - pos1);
  std::cout << "\n-------" << solverName<<" Solver -------" << std::endl;

  auto start = std::chrono::steady_clock::now();

  //核心部分
  TSolver solver;
  solver.compute(A);
  solver.setTolerance(1e-4); //|Ax-b|/|b|，默认是机器精度
  solver.setMaxIterations(1000);//默认是2*列数
  x = solver.solve(b);

  std::cout << "#iterations: " << solver.iterations() << ", estimated error: " << solver.error() << ",  tolerance: " << solver.tolerance() << std::endl;
  
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "solver elapsed seconds: " << elapsed_seconds.count() << std::endl;
}

int main()
{

  Eigen::SparseMatrix<double> S = Eigen::MatrixXd::Random(N, N).sparseView(0.5, 1);
  S = S.transpose() * S;

  MatrixReplacement A;
  A.attachMyMatrix(S);

  Eigen::VectorXd b(N), x(N);
  b.setRandom();

  std::cout << "A is " << N << "x" << N << std::endl;

  // Solve Ax = b using various iterative solver with matrix-free version:

  linSol<Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner>>(A, b, x);

  linSol<Eigen::BiCGSTAB<MatrixReplacement, Eigen::IdentityPreconditioner>>(A, b, x);

  linSol<Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner>>(A, b, x);

  linSol<Eigen::DGMRES<MatrixReplacement, Eigen::IdentityPreconditioner>>(A, b, x);

  linSol<Eigen::MINRES<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner>>(A, b, x);

  return 0;
}