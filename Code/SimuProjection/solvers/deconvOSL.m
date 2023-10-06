% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function recon = deconvOSL(forwardFUN, backwardFUN, img, iter, init, lambda)
% 'One step late' deconvolution algorithm
% When lambda == '0' this is the Richardson-Lucy algorithm
fprintf('\nDeconvolution:')

% Initialize volume
recon = init;

whiteImg = ones(size(img));
onesBack = backwardFUN(whiteImg);

mx = max(img(:));
if(mx > 1.0)
img = double(img)/double(mx); % normalize
else
img = double(img); % leave intact
end

% laplacian kernel
if (size(init, 3) > 1)
    h = -fspecial3('laplacian',0,0);
else
    h = -fspecial('laplacian',0);
end
    
% normalize filter
h = h./max(abs(h(:)));

for i = 1:iter
    tic

    div = imfilter(recon, h);
    regularizer = 1./(onesBack + lambda*div);
    
    % make sure the computations are safe
    regularizer(isnan(regularizer)) = 0;
    regularizer(isinf(regularizer)) = 0;

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
    recon = recon.*bpjError.*regularizer;
    
    ttime = toc;
disp(['  iter ' num2str(i) ' | ' num2str(iter) ', took ' num2str(ttime) ' secs']);
end
