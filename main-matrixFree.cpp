//matrix-free solver
#include <iostream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

class MatrixReplacement; // 提前声明一个包装类
using Eigen::SparseMatrix;

// 定义一个traits模板类
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

//定义一个包装类模板MatrixReplacement，这个类模板是核心。这个类用来包装sparse matrix
// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
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

    //MYADD 要更改的地方1：返回行号和列号的函数
  Index rows() const { return mp_mat->rows(); }
  Index cols() const { return mp_mat->cols(); }

    //MYADD 要更改的地方2：定义乘法运算符。
  template <typename Rhs>
  Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const
  {
    return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // Custom API:
  //默认构造函数，给个0值
  MatrixReplacement() : mp_mat(0) {}

    //真正把矩阵传进来的函数
  void attachMyMatrix(const SparseMatrix<double> &mat)
  {
    mp_mat = &mat;
  }
    
    //返回矩阵的函数
  const SparseMatrix<double> my_matrix() const { return *mp_mat; }

private:
    //被存储的矩阵数据的指针
  const SparseMatrix<double> *mp_mat;
};


//真正定义乘法运算符的地方
// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
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

        //MYADD: 乘以并加上
      template <typename Dest>
      static void scaleAndAddTo(Dest &dst, const MatrixReplacement &lhs, const Rhs &rhs, const Scalar &alpha)
      {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        assert(alpha == Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);

        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        // but let's do something fancier (and less efficient):
        for (Index i = 0; i < lhs.cols(); ++i)
          dst.noalias() += lhs.my_matrix() * rhs
          // dst += rhs(i) * lhs.my_matrix().col(i);
      }
    };

  }
}

int main()
{
  int n = 1000;
  Eigen::SparseMatrix<double> S = Eigen::MatrixXd::Random(n, n).sparseView(0.5, 1);
  S = S.transpose() * S;

  MatrixReplacement A;
  A.attachMyMatrix(S);

  Eigen::VectorXd b(n), x;
  b.setRandom();

  std::cout << "A is " << n << "x" << n << std::endl;
  // Solve Ax = b using various iterative solver with matrix-free version:
  auto start = std::chrono::steady_clock::now();
  {
    Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    cg.compute(A);
    x = cg.solve(b);
    std::cout << "CG:       #iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "CG elapsed seconds: " << elapsed_seconds.count() << std::endl;

  start = std::chrono::steady_clock::now();
  {
    Eigen::BiCGSTAB<MatrixReplacement, Eigen::IdentityPreconditioner> bicg;
    bicg.compute(A);
    x = bicg.solve(b);
    std::cout << "BiCGSTAB: #iterations: " << bicg.iterations() << ", estimated error: " << bicg.error() << std::endl;
  }
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "BiCGSTAB elapsed seconds: " << elapsed_seconds.count() << std::endl;

  start = std::chrono::steady_clock::now();
  {
    Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner> gmres;
    gmres.compute(A);
    x = gmres.solve(b);
    std::cout << "GMRES:    #iterations: " << gmres.iterations() << ", estimated error: " << gmres.error() << std::endl;
  }
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "GMRES elapsed seconds: " << elapsed_seconds.count() << std::endl;

  start = std::chrono::steady_clock::now();
  {
    Eigen::DGMRES<MatrixReplacement, Eigen::IdentityPreconditioner> gmres;
    gmres.compute(A);
    x = gmres.solve(b);
    std::cout << "DGMRES:   #iterations: " << gmres.iterations() << ", estimated error: " << gmres.error() << std::endl;
  }
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "DGMRES elapsed seconds: " << elapsed_seconds.count() << std::endl;

  start = std::chrono::steady_clock::now();
  {
    Eigen::MINRES<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> minres;
    minres.compute(A);
    x = minres.solve(b);
    std::cout << "MINRES:   #iterations: " << minres.iterations() << ", estimated error: " << minres.error() << std::endl;
  }
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end - start;
  std::cout << "MINRES elapsed seconds: " << elapsed_seconds.count() << std::endl;
}