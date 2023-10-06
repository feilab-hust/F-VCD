function [Camera,LensletGridModel] = Compute_camera(Config,superResFactor)
Camera = Config;
Camera.spacingPixels = Camera.spacingPixels * superResFactor;
Camera.superResFactor = superResFactor;

Camera.objRad = Camera.fobj * Camera.NA; % objective radius
Camera.k = 2*pi*Camera.n/Camera.WaveLength; % wave number
Camera.M = Camera.fm*Camera.f1/(Camera.f2*Camera.fobj); %total system magnification

% field stop radius 
Camera.fsRad = Camera.lensPitch/2 * Camera.f2/Camera.fm;

% field of view radius
Camera.fovRad =  Camera.fsRad * Camera.fobj/Camera.f1;

%%%% MLA array descriptor
% LensletGridModel.gridType = Camera.gridType;
% LensletGridModel.UMax = Camera.noLensHoriz;
% LensletGridModel.VMax = Camera.noLensVert;
% LensletGridModel.FirstPosShiftRow = Camera.shiftRow; % in hexagonal grids 
% LensletGridModel.Orientation = 'horz';


% LensletGridModel.HSpacing = LensletGridModel.HSpacing + mod(LensletGridModel.HSpacing,2); % make sure it is even

% if(strcmp(Camera.gridType, 'hex'))
%     LensletGridModel.VSpacing = round(sqrt(3)/2*LensletGridModel.HSpacing);
%     LensletGridModel.VSpacing = LensletGridModel.VSpacing + mod(LensletGridModel.VSpacing,2); % make sure it is even
% else
    LensletGridModel.HSpacing = Camera.spacingPixels; %the spacing between lenslet centers in pixels 
    LensletGridModel.VSpacing = LensletGridModel.HSpacing;
% end
%LensletGridModel.HOffset = Camera.horizOffset * superResFactor;
%LensletGridModel.VOffset = Camera.vertOffset * superResFactor;
% LensletGridModel.Rot = Camera.gridRot;