%% IRLS

% m-file qsm_irls.m to find the optimal solution to a qsm adaptation of Ax=b
%  Newton iterative update of solution, x, for  M > N.
%  For 2<p<infty, use homotopy parameter K = 1.01 to 2
%  For 0<p<2, use K = approx 0.7 - 0.9
%  csb 10/20/2012
function x = QSM_IRLS(A, At, b, solve, x, p, K, KK)
% defaults
if nargin < 5, KK=10;  end;
if nargin < 4, K = 1.5;  end;
if nargin < 3, p = 10; end;
E = [];
epsilon = .5;
% define basic parameters
xnum = numel(x);

for k = 1:KK
    % define errors
    e = A(x)-b;
    E = [E norm(e)];
%    disp('cut')
%    disp(e(1:2))
%    disp(e(xnum+1:xnum+2))

    % prevents weights from going to infinity
    for i = xnum+1:numel(e)
        if abs(e(i)) < epsilon
            e(i) = epsilon;
        end
    end

    % define weights
    w = vertcat(ones(xnum, 1), abs(e(xnum+1:end)).^((p-2)/2));

%    disp(w(1:2))
%    disp(w(xnum+1:xnum+2))
    W  = w;
%    disp(W(1:2))
%    disp(W(xnum+1:xnum+2))

    linearoperator = @(arg) flatten(At (W .* W .* A(arg)));
    constant = flatten(At(W .* b));

    % solve the weighted least squares problem
    x  = solve(linearoperator, constant, x);
end
