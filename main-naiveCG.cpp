// https://phtournier.pages.math.cnrs.fr/5mm29/cg/

#include <cmath>

// scalar product (u,v)
double operator,(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  double sp=0.;
  for(int j=0; j<u.size(); j++){sp+=u[j]*v[j];}
  return sp; 
}

// norm of a vector u
double Norm(const std::vector<double>& u) { 
  return sqrt((u,u));
}

// addition of two vectors u+v
std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  std::vector<double> w=u;
  for(int j=0; j<u.size(); j++){w[j]+=v[j];}
  return w;
}

// multiplication of a vector by a scalar a*u
std::vector<double> operator*(const double& a, const std::vector<double>& u){ 
  std::vector<double> w(u.size());
  for(int j=0; j<w.size(); j++){w[j]=a*u[j];}
  return w;
}

//addition assignment operator, add v to u
void operator+=(std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  for(int j=0; j<u.size(); j++){u[j]+=v[j];}
}

std::vector<double> CG(const MapMatrix& A, const std::vector<double>& b, double tol=1e-6) {

  assert(b.size() == A.NbCol());
  assert(A.NbCol()== A.NbRow());
  std::vector<double> x(b.size(),0.);

  double nr=Norm(b);
  double epsilon = tol*nr; 
  std::vector<double> p=b,r=b, Ap=A*p;
  double np2=(p,Ap), alpha=0.,beta=0.;

  int num_it = 0;
  while(nr>epsilon) {
    alpha = (nr*nr)/(np2);
    x += (+alpha)*p; 
    r += (-alpha)*Ap;
    nr = Norm(r);    
    beta = (nr*nr)/(alpha*np2); 
    p = r+beta*p;    
    Ap=A*p;
    np2=(p,Ap);

    num_it++;
    if(!(num_it%20)) {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << nr     << "\n";
    }
  }
  return x;
}