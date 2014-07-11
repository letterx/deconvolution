%% IRLS

% m-file qsm_irls.m to find the optimal solution to a qsm adaptation of Ax=b
%  Newton iterative update of solution, x, for  M > N.
%  For 2<p<infty, use homotopy parameter K = 1.01 to 2
%  For 0<p<2, use K = approx 0.7 - 0.9
%  csb 10/20/2012
function x = QSM_IRLS_new(A, At, b, solve, x, p, KK)
% defaults
if nargin < 5, KK=10;  end;
if nargin < 3, p = 10; end;
E = [];
epsilon = 10^-2;

% define basic parameters
xnum = numel(x);

% initial starting point
x  = solve(A, b, x);

for k = 1:KK
    % define errors
    e = A(x)-b;
    E = [E norm(e)];

    % keep weights from going to infnty
%    for i = 1:numel(e)
%        if (e(i) < epsilon)
%            e(i) = epsilon;
%        end
%    end

    % define weights
    w = flatten(abs(e).^((p-2)/2));
    W  = w/sum(w);

    disp('before cut')
    disp(norm(A(x)-b, p))
    disp(norm(W.*(A(x)-b), 2))

    % solve the weighted least squares problem
    linop = @(arg) At(W.*W.*A(arg));
    const = At(W.*W.*b);
    x  = solve(linop, const, x);

    disp('cut')
    disp(norm(A(x)-b, p))
    disp(norm(W.*(A(x)-b), 2))
    disp(e(1:3))
    disp(w(1:3))
    disp(W(1:3))
    disp(x(1:3))

end
