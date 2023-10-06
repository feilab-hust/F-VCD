% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu and Josue Page

%  This function implements the depth-dependent aliasing-aware
%  deconvolution for conventional LFM
%  For details: A. Stefanoiu et. al., "Artifact-free deconvolution in light field microscopy", Optics Express, 27(22):31644, (2019)

function reconVolume = deconvEMS(forwardFUN, backwardFUN, LFimage, iter, reconVolume, filterFlag, kernelFFT,  onesForward, onesBack)

for i = 1:iter
    tic;
    LFimageGuess = forwardFUN(reconVolume);
    
    errorLFimage = LFimage./LFimageGuess.*onesForward;
    errorLFimage(~isfinite(errorLFimage)) = 0;
    
    errorBack = backwardFUN(errorLFimage);
    errorBack = errorBack./onesBack;
    errorBack(~isfinite(errorBack)) = 0;

    % update
    reconVolume = reconVolume.*errorBack;
    
    % Artifacts correction step
    if(filterFlag)         
        for j = 1:size(errorBack,3)
            reconVolume(:,:,j) = abs(fftshift(ifft2(kernelFFT(:,:,j).*fft2(reconVolume(:,:,j)))));
        end
    end
    reconVolume(~isfinite(reconVolume)) = 0;
    ttime = toc;
    fprintf(['\niter ' num2str(i) ' | ' num2str(iter) ', took ' num2str(ttime) ' secs']);
end

