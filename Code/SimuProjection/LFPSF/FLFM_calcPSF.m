% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu 

% Adapted after Felix Wechsler's 'Wavefront Coding Techniques for Extended Depth of Field in Light Field Microscopy' BA thesis
% felix.wechsler@tum.de

function [psf] = FLFM_calcPSF(p1, p2, p3, Camera, Resolution)
% fobj -> MO's focal length
% NA -> MO's numerical aperture
% yspace, xspace -> sensor space coordinates

n = Camera.immersion_n; % refractive index
kn = 2*pi/Camera.WaveLength*n; % wave number

%% Wavefront at the MO's fron focal plane
A = 1; % some amplitude

% choose unevent number of points to have a mid point
N = 8191; % samples -> relatively large to make sure the field is not undersampled
midPoint = (N + 1) / 2;
NoutputY = length(Resolution.yspace);
NoutputX = length(Resolution.xspace);

% physical length of the sampled input field at the NOP in micrometers
LU0 = 200;

% coordinates of the NOP 
x = linspace(-LU0/2, LU0/2, N);
y = x;
[X,Y] = meshgrid(x,y);

% if the source point is at the front focal plane (NOP) => a delta peak
if p3 == 0
    p3 = 1e-8;
end

% distance from point source to the MO
r = sqrt((X-p1).^2.+(Y-p2).^2.+p3.^2);

% when p3>0, propagate back to the lens 
if p3 > 0
    r = -1*r;
end
% field at the NOP
U0 = -1i*A*kn/2/pi./r .* exp(1i.*kn.*r);

%% Wavefront at the MO's front focal plane
% due to the FFT there is a scaling factor and LU1 is the length of the field at the the back aperture stop of the MO
[U1, LU1] = FLFM_lensProp(U0, LU0, Camera.WaveLength, Camera.fobj);
coeffU1minus = -1i*exp(1i*kn*Camera.fobj)/Camera.WaveLength/Camera.fobj;
U1 = coeffU1minus.*U1;
% back aperture stop
circ = @(x,y,r) (x.^2.+y.^2.<r.^2); % pupil function
dobj = 2*Camera.fobj*Camera.NA;
U1 = U1.*circ(X./LU0.*LU1, Y./LU0.*LU1, dobj./2);

%% Relay magnification effect (when Camera.f1 != Camera.f2)
Mrelay = Camera.f2/Camera.f1;
% we have to reshape U1 to match the spacing and extent of the xspace and yspace
cut = [round(Resolution.yspace(end) / (Mrelay*LU1/2) * (N+1)/2), round(Resolution.xspace(end) / (Mrelay*LU1/2) * (N+1)/2)];
psf = U1(midPoint - cut(1) : midPoint + cut(1), midPoint - cut(2) : midPoint + cut(2));

% downsample the psf the back focal plane of the MO
psf = imresize(psf, [NoutputY NoutputX], 'bicubic');
