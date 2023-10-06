% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function recon = deconvRL(forwardFUN, backwardFUN, img, iter,init,cropped_size,save_path)
% Richardson-Lucy deconvolution algorithm
fprintf('\nDeconvolution:')

% Initialize volume
recon = init;

for i = 1:iter
    tic
    % simulate forward projection of the current reconstructed volume
    fpj = forwardFUN(recon);
    
    % compute error towards the real image
    errorBack = img./fpj;
    
    % make sure the computations are safe
    errorBack(isnan(errorBack)) = 0;
    errorBack(isinf(errorBack)) = 0;
    errorBack(errorBack < 0) = 0;
    
    % backproject the error
    bpjError = backwardFUN(errorBack);
    
    % update the result
    recon = recon.*bpjError;
    ttime = toc;
    fprintf(['\niter ' num2str(i) ' | ' num2str(iter) ', took ' num2str(ttime) ' secs']);
    if mod(i,10)==0 && i~=0
        recon = gather(recon);
        hs=ceil(size(recon,1)/2)-floor(cropped_size(1)/2);
        he=ceil(size(recon,1)/2)+floor(cropped_size(1)/2);

        ws=ceil(size(recon,2)/2)-floor(cropped_size(2)/2);
        we=ceil(size(recon,2)/2)+floor(cropped_size(2)/2);

        cropped_recon = recon(hs:he,ws:we,:);
        str_save=strrep(save_path,'.tif',sprintf('_ite%03d.tif',i));
        write3d(cropped_recon,str_save,32)
    end
end