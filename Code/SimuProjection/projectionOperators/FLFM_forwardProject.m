% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function Projection = FLFM_forwardProject(H, realSpace)
% forwardProjectFLFM: Forward projects a volume to a lenslet image by applying the LFPSF

Projection = zeros([size(realSpace,1), size(realSpace,2)]);
for j = 1:size(H,3)
    fprintf('[%d/%d] projection\n',j,size(H,3))
    Projection = Projection + conv2(realSpace(:,:,j), full(H{1,1,j}),'same');
end