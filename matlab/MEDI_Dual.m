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
%   ----optional----   
%   MEDI_L1('smv', radius,...) - specify the radius for the spherical mean
%                                value operator using differential form
%   MEDI_L1('merit',...) - turn on model error reduction through iterative
%                          tuning
%   MEDI_L1('zeropad',padsize,...) - zero pad the matrix by padsize
%
%   When using the code, please cite 
%   T. Liu et al. MRM 2013;69(2):467-76
%   J. Liu et al. Neuroimage 2012;59(3):2560-8.
%   T. Liu et al. MRM 2011;66(3):777-83
%   de Rochefort et al. MRM 2010;63(1):194-206
%
%   Adapted from Ildar Khalidov
%   Modified by Tian Liu on 2011.02.01
%   Modified by Tian Liu and Shuai Wang on 2011.03.15
%   Modified by Tian Liu and Shuai Wang on 2011.03.28 add voxel_size in grad and div
%   Last modified by Tian Liu on 2013.07.24

function [x, D, m, RDF, cost_reg_history, cost_data_history] = MEDI_Dual(varargin)

[lambda iFreq RDF N_std iMag Mask matrix_size matrix_size0 voxel_size delta_TE CF B0_dir merit smv radius data_weighting gradient_weighting Debug_Mode] = parse_QSM_input(varargin{:});

fprintf('Begin MEDI_Dual\n');

%%%%%%%%%%%%%%% weights definition %%%%%%%%%%%%%%
cg_max_iter = 100;
cg_tol = 0.01;
max_iter = 10;
tol_norm_ratio = 0.1;
data_weighting_mode = data_weighting;
gradient_weighting_mode = gradient_weighting;
grad = @cgrad;
div = @cdiv;

tempn = double(N_std);
D=dipole_kernel(matrix_size, voxel_size, B0_dir);

if (smv)
    S = SMV_kernel(matrix_size, voxel_size,radius);
    Mask = SMV(Mask, matrix_size,voxel_size, radius)>0.999;
    D=S.*D;
    RDF = RDF - SMV(RDF, matrix_size, voxel_size, radius);
    RDF = RDF.*Mask;
    tempn = sqrt(SMV(tempn.^2, matrix_size, voxel_size, radius)+tempn.^2);
end

m = dataterm_mask(data_weighting_mode, tempn, Mask);
b0 = m.*exp(1i*RDF);




iter=0;
x = zeros(matrix_size); %real(ifftn(conj(D).*fftn((abs(m).^2).*RDF)));
if (~isempty(findstr(upper(Debug_Mode),'SAVEITER')))
    store_CG_results(x/(2*pi*delta_TE*CF)*1e6.*Mask);%add by Shuai for save data
end
res_norm_ratio = Inf;
cost_data_history = zeros(1,max_iter);
cost_reg_history = zeros(1,max_iter);

tic

H  = @(arg) m.*(real(ifftn(D.*fftn(arg))));
Ht = @(arg) real(ifftn(D.*fftn(arg.*m)));

function progress(x, dual, primalData, primalReg, smoothing)
    iter = iter+1;
    time = toc;
    fprintf('Iteration %d\t dual: %4.2f\tData: %4.2f\tReg: %4.2f\tSmoothing: %4.2f\ttime %4.2f\n', iter, dual, primalData, primalReg, smoothing, time);
    if (iter > 100)
        error('maxiter reached')
    end
end

deconvolveParams = struct();
deconvolveParams.progress = @progress;
deconvolveParams.maxIter = 100;
deconvolveParams.dataSmoothing = 0.001;
deconvolveParams.smoothing = 100;
deconvolveParams.minSmoothing = 1;
deconvolveParams.smoothWeight = 10;
deconvolveParams.smoothMax = 100.0;

fprintf('Begin deconvolveDual\n');
x = deconvolveDual(H, Ht, m.*RDF, deconvolveParams);
fprintf('End deconvolveDual\n');

wres = m.*(real(ifftn(D.*fftn(x))) - RDF);
fprintf('Final data term: %f\n', norm(wres(:),2)^2);

%convert x to ppm
% x = x/(2*pi*delta_TE*CF)*1e6.*Mask;

if (matrix_size0)
    x = x(1:matrix_size0(1), 1:matrix_size0(2), 1:matrix_size0(3));
    iMag = iMag(1:matrix_size0(1), 1:matrix_size0(2), 1:matrix_size0(3));
    RDF = RDF(1:matrix_size0(1), 1:matrix_size0(2), 1:matrix_size0(3));
    Mask = Mask(1:matrix_size0(1), 1:matrix_size0(2), 1:matrix_size0(3));
    matrix_size = matrix_size0;
end

store_QSM_results(x, iMag, RDF, Mask,...
                  'Norm', 'L1','Method','MEDIN','Lambda',lambda,...
                  'SMV',smv,'Radius',radius,'IRLS',merit,...
                  'voxel_size',voxel_size,'matrix_size',matrix_size,...
                  'Data_weighting_mode',data_weighting_mode,'Gradient_weighting_mode',gradient_weighting_mode,...  
                  'L1_tol_ratio',tol_norm_ratio, 'Niter',iter,...
                  'CG_tol',cg_tol,'CG_max_iter',cg_max_iter,...
                  'B0_dir', B0_dir);

end





              
