## Intro
这是测试Eigen的练习项目。

将每个main_xxx.cpp文件改为main.cpp，然后编译运行即可。

目前测试了
- BiCGStab(一种迭代求解器)
- Cholesky(一种分解求解器)
- fillSparseMatrix(填充稀疏矩阵)
- traverseSparseMatrix(遍历稀疏矩阵)
- matrixFree(无矩阵迭代求解器)
- SparseMatrix1(稀疏矩阵求解Laplace问题的案例, 采用SparseCholesky)

## data
data文件夹下面的是matrix market格式的矩阵数据。

数据来源：

https://math.nist.gov/MatrixMarket/data/SPARSKIT/drivcav/drivcav.html

(gz格式请使用gzip -d xxx.gz解压)