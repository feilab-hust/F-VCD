function [imageStack,BitDepth] = imread3d(file)
%IMREAD3D Reads 3D TIFF stack
%   Inputs:
%       file - file name of TIFF stack

    tiffInfo = imfinfo(file);
%     assert(tiffInfo(1).BitDepth == 8, sprintf('%s is not in 8-bit\n convert this image into 8-bit in ImageJ first\n', tiffInfo(1).Filename))
    BitDepth=tiffInfo(1).BitDepth;
    Width = tiffInfo(1).Width;
    Height = tiffInfo(1).Height;
    Depth = numel(tiffInfo);
    if BitDepth==16
        type='uint16';
    else
         type='double';
    end
    imageStack = zeros(Height, Width, Depth,type);  % double by default
    for i = 1:Depth
        %imageStack(:,:,i) = imread(file, 'Index', i, 'Info', tiffInfo);
        imageStack(:,:,i) = imread(file, i);
    end
    
    %imageStack = uint8(imageStack);
end

