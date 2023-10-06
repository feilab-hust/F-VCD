% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function BackProjection = FLFM_backwardProject(Ht, projection)
% backwardProjectFLFM: back projects a lenslet image into a volume.

BackProjection = zeros([size(projection,1), size(projection,2), size(Ht,3)]);
for j = 1:size(Ht,3)
    fprintf('[%d/%d] BACKprojection\n',j,size(Ht,3))
    BackProjection(:,:,j) = conv2(projection , full(Ht{1,1,j}),'same');
end