
%% Define FLFM_Projection
 function [] = FLFM_Projection(Config,superResFactor,realSpace,DirPath,FileName,dynamic_range)
    [Camera,LensletGridModel] = Compute_camera(Config,superResFactor);
    Resolution = Compute_Resolution(Camera,LensletGridModel);
    save('resolution.mat','Resolution');
    [H, Ht] = FLFM_computeLFMatrixOperators(Camera, Resolution);
%     save()
    Projection = FLFM_forwardProject(H, realSpace);
    Projection = uint16(Projection .* dynamic_range);
%     LF_dir = fullfile(DirPath, 'LF');
%     if ~exist(LF_dir, 'dir')
%         mkdir(LF_dir);
%     end
%     tic
    imwrite(Projection,[LF_dir,'LF_',FileName]);
%     toc
%     imwrite(Projection,FileName);
 end