clc
clear
addpath ./util
addpath ./solvers
addpath ./LFPSF
addpath ./projectionOperators

%% loading H
% H=psf;
ds_z=0
GPU_enable=1;
cropped_size=[499,499];
root_dir='G:\FLFM_SIMU\hex_shujia\psf_simu\m3.4-3.4';
H=load(fullfile(root_dir,'H_WF.mat'));
H=H.H;

Ht=load(fullfile(root_dir,'Ht_WF.mat'));
Ht=Ht.Ht;
H_size=size(full(H{:,:,1}));

[hr_file_name,hr_filepath] = uigetfile('*.tif','Select HR Volumes','MultiSelect','on');
if ~iscell(hr_file_name)
    hr_file_name = {hr_file_name};
end

file_name=sprintf('\\Deconv');
save_Path=fullfile(hr_filepath,file_name);
if exist(save_Path,'dir')==7
    ;
else
    mkdir(save_Path);
end

for img_idx=1:length(hr_file_name)
    img_name=hr_file_name{img_idx};
    img_path=fullfile(hr_filepath,img_name);
    LensletImage=double(imread3d(img_path));
    volumeSize = [size(LensletImage,1), size(LensletImage,2), size(Ht,3)];
    init = ones(volumeSize);
    %%
    if ~GPU_enable
        forwardFUN = @(volume) FLFM_forwardProject(H, volume);
        backwardFUN = @(projection) FLFM_backwardProject(Ht, projection);
    else 
        forwardFUN = @(volume) FLFM_forwardProjectGPU(H, volume);
        backwardFUN = @(projection) FLFM_backwardProjectGPU(Ht, projection);
        global zeroImageEx;
        global exsize;
        xsize = [volumeSize(1), volumeSize(2)];
        msize = [H_size(1), H_size(2)];
        mmid = floor(msize/2);
        exsize = xsize + mmid;  
        exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];    
        zeroImageEx = gpuArray(zeros(exsize, 'single'));
        disp(['FFT size is ' num2str(exsize(1)) 'X' num2str(exsize(2))]); 
    end
    
    %%  Richardson Lucy deconvolution
    
    
    iter  = 30; % number of iterations
    savePath=fullfile(save_Path,sprintf('Recon_%s.tif',img_name));
    recon = deconvRL(forwardFUN, backwardFUN, LensletImage, iter, init,cropped_size,savePath);
end



