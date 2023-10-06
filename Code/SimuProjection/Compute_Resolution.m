function Resolution = Compute_Resolution(Camera,LensletGridModel)
%% Compute sensor resolution
NspacingLenslet = [LensletGridModel.VSpacing, LensletGridModel.HSpacing]; % Number of pixels behind a lenslet 

% Corresponding sensor resolution
% if(strcmp(Camera.gridType, 'hex'))
%     sensorRes = [Camera.lensPitch*cosd(30)/NspacingLenslet(1), Camera.lensPitch./NspacingLenslet(2)];
%     Nnum = [max(NspacingLenslet), max(NspacingLenslet) ];
% end
% if(strcmp(Camera.gridType, 'reg'))
sensorRes = [Camera.lensPitch/NspacingLenslet(1), Camera.lensPitch./NspacingLenslet(2)];
Nnum = NspacingLenslet;
% end
Nnum = Nnum - (1-mod(Nnum,2));

%% Object space resolution (voxel size in um) 
texRes = sensorRes./Camera.M;
texRes(3) = Camera.depthStep;
% field of view in voxels
Resolution.fovRadVox = [round(Camera.fovRad./texRes(1)), round(Camera.fovRad./texRes(2))];
%% Set up a struct containing the resolution related info
Resolution.Nnum = Nnum;
Resolution.sensorRes = sensorRes;
Resolution.texRes = texRes;
Resolution.depthStep = Camera.depthStep;
Resolution.depthRange = Camera.depthRange;
Resolution.depths = Camera.depthRange(1) : Camera.depthStep : Camera.depthRange(2);
Resolution.superResFactor = Camera.superResFactor;
%% compute LensletCenters
% Xposcenters = [Camera.X_center-LensletGridModel.HSpacing/2*Camera.dx : Camera.dx : Camera.X_center+LensletGridModel.HSpacing/2*Camera.dx];
% Yposcenters = [Camera.Y_center-LensletGridModel.VSpacing/2*Camera.dy : Camera.dy : Camera.Y_center+LensletGridModel.VSpacing/2*Camera.dy]';
% Lenslet_xposcenters = repelem([Xposcenters;],LensletGridModel.VSpacing);
% Lenslet_yposcenters = repelem([Yposcenters],LensletGridModel.HSpacing);
Xposcenters = [Camera.X_center-(Camera.MLAnumX-1)*(LensletGridModel.HSpacing)/2 : LensletGridModel.HSpacing : Camera.X_center+(Camera.MLAnumX-1)*(LensletGridModel.HSpacing)/2];%[110,324,538,752,966]
Yposcenters = [Camera.Y_center-(Camera.MLAnumY-1)*LensletGridModel.VSpacing/2 : LensletGridModel.VSpacing : Camera.Y_center+(Camera.MLAnumY-1)*LensletGridModel.VSpacing/2]';
Lenslet_xposcenters = repmat(Xposcenters,Camera.MLAnumY,1);
Lenslet_yposcenters = repmat(Yposcenters,1,Camera.MLAnumX);       

Resolution.LensletCenters(:,:,1) = ceil(Lenslet_xposcenters);
Resolution.LensletCenters(:,:,2) = ceil(Lenslet_yposcenters);
%% Sensorsize in pixels
Resolution.sensorSize = Camera.SensorSize;
