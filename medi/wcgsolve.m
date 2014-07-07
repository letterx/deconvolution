% wcgsolve.m
%
% Solve a symmetric positive definite system Ax = b with weights via conjugate gradients.
%
% Assumes that A is a function handle
%
% http://en.wikipedia.org/wiki/Conjugate_gradient_method
function x = wcgsolve(A, b, w, tol, maxiter, verbose, x0)

matrix_size=size(b);

x = x0;
r = w.*(b-A(x));
d = r;

delta = r'*r;
delta0 = b'*b;
numiter = 0;
bestx = x;
bestres = sqrt(delta/delta0); 
while ((numiter < maxiter) & (delta > tol^2*delta0))
    q = w.*A(d);
    
    alpha = delta/(d'*q);
    x = x + alpha*d;

    if (mod(numiter+1,50) == 0)
    r = w.* (b - reshape(A(reshape(x,matrix_size)),size(b)));
    else
    r = r - alpha*q;
    end

    deltaold = delta;
    delta = r'*r;
    beta = delta/deltaold;

    d = r + beta*d;

    numiter = numiter + 1;
    if (sqrt(delta/delta0) < bestres)
    bestx = x;
    bestres = sqrt(delta/delta0);
    end    

    if ((verbose) & (mod(numiter,verbose)==0))
    disp(sprintf('cg: Iter = %d, Best residual = %8.3e, Current residual = %8.3e', ...
      numiter, bestres, sqrt(delta/delta0)));
    end
end

if (verbose)
  disp(sprintf('cg: Iterations = %d, best residual = %14.8e', numiter, bestres));
end

res = bestres;
iter = numiter;

