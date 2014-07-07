% flatten.m
%
% Default way of turning a matrix into a vector. Creates horizontal vectors.
function x = flatten(arg)
x = arg(:)';
