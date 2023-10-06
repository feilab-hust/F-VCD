%%
%% This FLFM PSF simulation code is used to generate the FLFM projections for F-VCD training, which is modified from olaf[1].
%% [1] Stefanoiu, A., Page, J. & Lasser, T. "olaf: A flexible 3d reconstruction framework for light field microscopy".  (2019).
%% 
import2ws();
addpath ./util
clc;
clear all;
% Import configurations of FLFM setup
psf_save_folder='View3_simu_[m3.4-3.4]_zstep_0.17';
Config.gridType =  'Reg';
Config.NA =1.4;
Config.M=100;
Config.f1 = 180000;
Config.fobj = Config.f1/Config.M;         %focal length of the objective lens. Camera.fobj = Camera.ftl/Camera.M

Config.f2 = 300000;        
Config.fm =120000;           %fm - focal length of the micro-lens.
Config.mla2sensor = Config.fm;   %distance between the MLA plane and camera sensor plane.

Config.lensPitch = 3250;
Config.WaveLength = 510*1e-3;

Config.spacingPixels=499;  %the number of pixels between two horizontally neighboring lenslets.
Config.pixelPitch =Config.lensPitch /Config.spacingPixels; % sensor pixel pitch (in ?m).

Config.immersion_n=1.518;
Config.n = 1;
Config.SensorSize = [1497,1497]; %the size of the input light field image,1075
Config.X_center = ceil(Config.SensorSize(1)/2);
Config.Y_center = ceil(Config.SensorSize(2)/2);
Config.depthStep = 0.17;
Config.depthRange = [-3.4,3.4];

Config.MLAnumX = 3; % deprecated 
Config.MLAnumY = 3; % deprecated 

theta_dof= Config.lensPitch/sqrt(3)/Config.f2;
D_pupuil=Config.f2*Config.NA*2/Config.M;
occupy_ratio=(Config.MLAnumX*Config.lensPitch)/D_pupuil;
FOV=Config.lensPitch*Config.f2/Config.fm/Config.M;
system_magnification=Config.f1/Config.fobj*Config.fm/Config.f2;
rxy=Config.WaveLength*Config.fm/Config.lensPitch/system_magnification;
rz = 1/(Config.M)^2*1/tan(theta_dof)*Config.WaveLength*Config.f2/Config.lensPitch;

sr_=rxy/(Config.pixelPitch/system_magnification);
N_=rxy*2.*Config.NA/Config.WaveLength;

tan_theta=Config.lensPitch/Config.f2;
DOF_ge_ideal=Config.lensPitch*Config.f2/Config.fm/(tan_theta*Config.M^2);
DOF_wave_ideal=N_^2*Config.WaveLength/Config.NA^2*(1+1/(2*sr_));

fprintf('[!]System Magnification: %.4f Optical Resolution (Rxy):%.4f (Rz): %.4f Voxel_size:(%.3f,%.3f)\n',system_magnification,rxy,rz,Config.pixelPitch/system_magnification,Config.depthStep);
fprintf('[!]System fov:%.3f DOF:%.3f D_pupuil:%.1f occupy_ratio:%.2f\n',FOV,DOF_wave_ideal,D_pupuil,occupy_ratio);
fprintf('[!]System Nnum:%d mla_num (%d-%d) Pitch size: %.2f\n',Config.spacingPixels,Config.MLAnumY,Config.MLAnumX,Config.pixelPitch);


superResFactor =1; 
[Camera,LensletGridModel] = Compute_camera(Config,superResFactor);
Resolution = Compute_Resolution(Camera,LensletGridModel);

if ~exist(psf_save_folder,'dir')
    mkdir(psf_save_folder);
end

[H, Ht] = FLFM_computeLFMatrixOperators(Camera, Resolution,Config.fm,psf_save_folder);



