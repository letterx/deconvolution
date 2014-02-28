//*****************************************************************
// Iterative template routine -- CG
//
// CG solves the symmetric positive definite linear
// system Ax=b using the Conjugate Gradient method.
//
// CG follows the algorithm described on p. 15 in the 
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//  
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//  
//*****************************************************************

template < class Matrix, class Vector, class Preconditioner, class Real >
int 
CG(const Matrix &A, Vector &x, const Vector &b,
   const Preconditioner &M, int &max_iter, Real &tol)
{
  Real resid;
  Vector p, z, q;
  typedef decltype(x(0)) Scalar;
  Scalar alpha{};
  Scalar beta{};
  Scalar rho{};
  Scalar rho_1{};

  Real normb = norm(b);
  Vector r = b - A*x;

  if (normb == 0.0) 
    normb = 1;
  
  if ((resid = norm(r) / normb) <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  for (int i = 1; i <= max_iter; i++) {
    z = M.solve(r);
    rho = dot(r, z);
    
    if (i == 1)
      p = z;
    else {
      beta = rho / rho_1;
      p = z + beta * p;
    }
    
    q = A*p;
    alpha = rho / dot(p, q);
    
    x += alpha * p;
    r -= alpha * q;
    
    if ((resid = norm(r) / normb) <= tol) {
      tol = resid;
      max_iter = i;
      return 0;     
    }

    rho_1 = rho;
  }
  
  tol = resid;
  return 1;
}

