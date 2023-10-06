% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu 

% Adapted after Felix Wechsler's 'Wavefront Coding Techniques for Extended Depth of Field in Light Field Microscopy' BA thesis
% felix.wechsler@tum.de

function [u2, L2] = FLFM_lensProp(u1, L1, lambda, z)
%% this function is based on the function propFF from the book "Computational Fourier Optics. A MATLAB Tutorial", D. Voelz, (2011).
% assumes uniform sampling

% u1 - source plane field
% L1 - source plane side length
% lambda - wavelength
% z - propagation distance
% L2 - observation plane side length
% u2 - observation plane field

% get input field size (number of sampling points)
[M, ~] = size(u1);

% source sample rate
dx1 = L1(1)/M;

% obs field sidelength
L2 = lambda*z./dx1;

u1(isnan(u1)) = 0;

% output field
% for odd length this
u2 = fftshift(fft2(ifftshift(u1))).*dx1.^2;

%this works for even length
%u2 = ifftshift(fft2(fftshift(u1))).*dx1.^2;
end
