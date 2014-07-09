% flatten.m
%
% Default way of turning a matrix into a vector. Creates vertical vectors.
function x = flatten(arg, align)
if nargin < 2, align = 0; end % default to vertical

if align == 0
    x = arg(:);
else
    x = reshape(arg, 1, []);
end
