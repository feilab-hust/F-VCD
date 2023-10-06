function Backprojection = FLFM_backwardProjectGPU(Ht, projection)
global zeroImageEx;
global exsize;

x3length = size(Ht,3);
Backprojection = gpuArray.zeros(size(projection, 1), size(projection, 2), x3length , 'single');
projection= gpuArray(projection);
for cc=1:x3length
    Hts = gpuArray(squeeze(full(Ht{1,1,cc})));
    Backprojection(:,:,cc) = Backprojection(:,:,cc) + conv2FFT(projection, Hts);
end

