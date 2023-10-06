function projection = FLFM_forwardProjectGPU( H, realspace )
global zeroImageEx;
global exsize;


projection = gpuArray.zeros(  size(realspace,1),   size(realspace,2), 'single');
zeroSlice=gpuArray.zeros(  size(realspace,1),   size(realspace,2), 'single');
for cc=1:size(realspace,3),
    Hs = gpuArray(squeeze(full(H{1,1,cc})));
    tempSlice = zeroSlice;
    tempSlice(:,:) = realspace( :, :,cc);
    projection = projection + conv2FFT(tempSlice, Hs);
end

