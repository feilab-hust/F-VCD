% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu 

function MLARRAY = FLFM_mlaTransmittance(Resolution, ulensPattern)

%% Compute the ML array as a grid of phase/amplitude masks corresponding to mlens
ylength = length(Resolution.yspace);
xlength = length(Resolution.xspace);

% activate lenslet centers -> set to 1

MLcenters = zeros(ylength, xlength);

center=ceil(size(MLcenters,1)/2);
d1=0;
d=size(ulensPattern,1);

MLcenters(center-ceil((d+d1)/2/sqrt(3)),center-ceil((d+d1)/2))=1;
MLcenters(center-ceil((d+d1)/2/sqrt(3)),center+ceil((d+d1)/2))=1;
% MLcenters(center,center)=1;
MLcenters(center+ceil((d+d1)/sqrt(3)),center)=1;

% apply the mlens pattern at every ml center
MLARRAY  = conv2(MLcenters, ulensPattern, 'same');

