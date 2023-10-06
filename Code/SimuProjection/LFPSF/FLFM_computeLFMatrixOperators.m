% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function [H, Ht] = FLFM_computeLFMatrixOperators(Camera, Resolution,ax2ca,psf_save_folder)

%% Sensor and ML space coordinates
IMGsizeHalf_y = floor(Resolution.sensorSize(1)/2);
IMGsizeHalf_x = floor(Resolution.sensorSize(2)/2);
Resolution.yspace = Resolution.sensorRes(1)*linspace(-IMGsizeHalf_y, IMGsizeHalf_y, Resolution.sensorSize(1));  %sensor plane coordinates
Resolution.xspace = Resolution.sensorRes(2)*linspace(-IMGsizeHalf_x, IMGsizeHalf_x, Resolution.sensorSize(2));
Resolution.yMLspace = Resolution.sensorRes(1)* [- ceil(Resolution.Nnum(1)/2) + 1 : 1 : ceil(Resolution.Nnum(1)/2) - 1];   %local lenslet coordinates
Resolution.xMLspace = Resolution.sensorRes(2)* [- ceil(Resolution.Nnum(2)/2) + 1 : 1 : ceil(Resolution.Nnum(2)/2) - 1];

%% Compute LFPSF operators

% compute the wavefront distribution incident on the MLA for every depth
fprintf('\nCompute the PSF stack at the back aperture stop of the MO.')
psfSTACK = FLFM_calcPSFAllDepths(Camera, Resolution);

% compute LFPSF at the sensor plane
fprintf('\nCompute the LFPSF stack at the camera plane:\n')
tolLFpsf = 0.001; % clap small valueds in the psf to speed up computations


for ii=1:length(ax2ca)
    d_camera=ax2ca(ii);
    [H, Ht] = FLFM_computeLFPSF(psfSTACK, Camera, Resolution, tolLFpsf,d_camera);
    save(fullfile(psf_save_folder,'H.mat'),'H','-v7.3');
    save(fullfile(psf_save_folder,'Ht.mat'),'Ht','-v7.3');
    [h,w]=size(H{1,1,1});
    psf_img=zeros(h,w,length(Resolution.depths));
    for i = 1:length(Resolution.depths)
        psf_img(:,:,i)=H{1,1,i};
    end
    psf_wf= abs(double(psfSTACK).^2);
    write3d(psf_wf,fullfile(psf_save_folder,'View3_f300_[m3.4-3.4]_Pupil.tif'),32);
    write3d(psf_img,fullfile(psf_save_folder,'View3_f300_[m3.4-3.4]_FLFM.tif'),32);
end