%% IRLS

% m-file qsm_irls.m to find the optimal solution to a qsm adaptation of Ax=b
%  Newton iterative update of solution, x, for  M > N.
%  For 2<p<infty, use homotopy parameter K = 1.01 to 2
%  For 0<p<2, use K = approx 0.7 - 0.9
%  csb 10/20/2012
function x = QSM_IRLS(A, solve, x, p, K, KK)
% defaults
if nargin < 5, KK=10;  end;
if nargin < 4, K = 1.5;  end;
if nargin < 3, p = 10; end;
pk = 2;                                      % Initial homotopy value L_2 solution
E = [];

% define basic parameters
x0 = zeros(size(x));
xlen = numel(x);

for k = 1:KK                                 % Iterate
    if p >= 2, pk = min([p, K*pk]);           % Homotopy change of p
      else pk = max([p, K*pk]); end

    % define errors
    e = A(x);

    b = zeros(size(e));

    % define weights
    w = vertcat(ones(xlen, 1), abs(e(xlen+1:end)).^((pk-2)/2));
    W  = reshape(w./sum(w), size(e));

    % solve the weighted least squares problem
    x1  = solve(A, b, W, x0);

    % update rules and records
    q  = 1/(pk-1);                            % Newton's parameter
    if p > 2, x = q*x1 + (1-q)*x; nn=p;       % partial update for p>2
      else x = x1; nn=2; end                 % no partial update for p<2
    ee = norm(e,nn);   E = [E ee];            % Error at each iteration
end

disp(E);
