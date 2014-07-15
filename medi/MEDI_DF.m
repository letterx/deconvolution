% Morphology Enabled Dipole Inversion (MEDI)
%   [x, cost_reg_history, cost_data_history] = MEDI_L1(varargin)
%
%   output
%   x - the susceptibility distribution 
%   cost_reg_history - the cost of the regularization term
%   cost_data_history - the cost of the data fidelity term
%   
%   input
%   RDF.mat has to be in current folder.  
%   MEDI_L1('lambda',lam,...) - lam specifies the regularization parameter
%                               lam is in front of the data fidelity term
%

function [x, cost_reg_history, cost_data_history] = MEDI_DF(varargin)

[lambda iFreq RDF N_std iMag Mask matrix_size matrix_size0 voxel_size delta_TE CF B0_dir merit smv radius data_weighting gradient_weighting Debug_Mode] = parse_QSM_input(varargin{:});


%%%%%%%%%%%%%%% weights definition %%%%%%%%%%%%%%
cg_max_iter = 100;
cg_tol = 0.01;
max_iter = 10;
tol_norm_ratio = 0.1;
data_weighting_mode = data_weighting;
gradient_weighting_mode = gradient_weighting;

tempn = double(N_std);
D=dipole_kernel(matrix_size, voxel_size, B0_dir);

%% Basic functions
grad = @(arg) cgrad(reshape(arg, matrix_size), voxel_size);
gm = grad(iMag);
gradsize = size(gm);
gradnum = numel(gm);

div = @(arg) cdiv(reshape(arg, gradsize), voxel_size);

solve = @(A, b, x0) real(cgsolve(A, b, cg_tol, cg_max_iter, 0, x0));
mask = dataterm_mask(data_weighting_mode, tempn, Mask);

H  = @(arg) flatten(mask.*(real(ifftn(D.*fftn(reshape(arg, matrix_size)))))); 

%% IRLS settings
rdf = flatten(RDF);
p = .8;
KK = 10;

%% Setup constants and linear operators

xnum = prod(matrix_size);

A = @(arg) vertcat(H(arg), flatten(grad(arg)));

At = @(arg) H(arg(1:xnum)) + ...
flatten(div(arg(xnum+1:xnum+gradnum)));

gz = zeros(numel(grad(RDF)), 1);
b = vertcat(rdf, gz);

%% IRLS
x = zeros(xnum, 1);
IR = QSM_IRLS_new(A, At, b, solve, x, p, KK);

% convert to ppm
ppm = @(x) reshape(x, size(Mask))/(2*pi*delta_TE*CF)*1e6.*Mask;

%assignin('base', 'TE', ppm(TE));
%assignin('base', 'SQ', ppm(SQ));
assignin('base', 'IR', reshape(IR, size(Mask))/1e2);
assignin('base', 'RDF', RDF);
assignin('base', 'Mask', Mask);
%convert x to ppm
x = ppm(x);
store_QSM_results(x, iMag, RDF, Mask,...
                  'Norm', 'L1','Method','MEDIN','Lambda',lambda,...
                  'SMV',smv,'Radius',radius,'IRLS',merit,...
                  'voxel_size',voxel_size,'matrix_size',matrix_size,...
                  'Data_weighting_mode',data_weighting_mode,'Gradient_weighting_mode',gradient_weighting_mode,...  
                  'L1_tol_ratio',tol_norm_ratio,...
                  'CG_tol',cg_tol,'CG_max_iter',cg_max_iter,...
                  'B0_dir', B0_dir);

end
