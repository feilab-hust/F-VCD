%% Initialization
clear all;
clc;
addpath ./util/
%% user setting
lr_path='..\..\Data\Substacks\NoiseLF';
hr_path='..\..\Data\Substacks\CleanLF';
GT_path='..\..\Data\Substacks\Stackx3';

bitdepth=16;
LF_cropped_size = 160;       
SR_factor=1;                     
rescale_factor = 1/3;             % stacksb  factor

overlap = 0.25;            
pixel_ratio=0.7; 
var_threshold=4e4;
assert(mod(LF_cropped_size,rescale_factor)==0,'cropped size cant be dived by factor');
cropped_size=LF_cropped_size/rescale_factor;
LR_cropped_size=LF_cropped_size/SR_factor;


save_all=false;
is_scale=false;
wrap_content=false;
% is_shuffle=true;
%% gen folder
save_root=strsplit(hr_path,'\');
temp_str=[];
for i=1:length(save_root)-1
    temp_str=strcat(temp_str,'\',save_root(i));
end
save_root=temp_str{1}(2:end);
folder_name=sprintf('Mito_View3_[LF%d_SF%d_WF%d]',LR_cropped_size,LF_cropped_size,cropped_size);
SavePath=fullfile(save_root,folder_name);

lr_SavePath = fullfile(SavePath,'LR');
hr_SavePath=fullfile(SavePath,sprintf('HR'));
GT_SavePath=fullfile(SavePath,sprintf('Stack'));

if exist(lr_SavePath, 'dir')==0
    mkdir(lr_SavePath);
end

if exist(hr_SavePath, 'dir')==0
    mkdir(hr_SavePath);
end

if exist(GT_SavePath, 'dir')==0
    mkdir(GT_SavePath);
end

%% Training data generation

lr_sourceDatasets = dir(lr_path);
lr_sourceDatasets(1:2) = [];
lr_Num = length(lr_sourceDatasets);

hr_sourceDatasets = dir(hr_path);
hr_sourceDatasets(1:2) = [];
hr_Num = length(hr_sourceDatasets);

GT_sourceDatasets = dir(GT_path);
GT_sourceDatasets(1:2) = [];
GT_Num = length(GT_sourceDatasets);


assert(lr_Num==hr_Num,'the numbers of image pair is not equal');
cropped_num=0;

for DatasetIndex = 1 : hr_Num
    lr_name=lr_sourceDatasets(DatasetIndex).name;
    hr_name=hr_sourceDatasets(DatasetIndex).name;
    gt_name=GT_sourceDatasets(DatasetIndex).name;

    fprintf('[%d/%d] processing %s\n',DatasetIndex,hr_Num,hr_name);

    [lr_img,bitdepth]=imread3d(fullfile(lr_path,lr_name));
    [hr_img,~]=imread3d(fullfile(hr_path,hr_name));
    [gt_img,~]=imread3d(fullfile(GT_path,gt_name));


    stack_p=prctile(gt_img(:), 99.9);
    local_thresh= pixel_ratio * stack_p;

    [height, width,depth] = size(gt_img);

    if height<cropped_size || width<cropped_size
        continue
    end

    overlap_px=floor(cropped_size*overlap);
    step= cropped_size - overlap_px;

    % cut image
    cropped_num_current = 0;
    idx_y = 0;
    h_wrap_flag=1;
    for i = 1 : step : height
        if i > height-cropped_size+1
            if wrap_content && h_wrap_flag
                i = height - cropped_size + 1;
                h_wrap_flag=0;
            else
                break;
            end
        end
        idx_x = 0;
        idx_y = idx_y + 1;
        w_wrap_flag=1;
        for j = 1 : step : width
            if j > width-cropped_size+1
                if wrap_content && w_wrap_flag
                    j = width - cropped_size + 1;
                    w_wrap_flag=0;
                else
                    break;
                end
            end

            idx_x = idx_x + 1;


            h = i + cropped_size - 1;
            w = j + cropped_size - 1;      %%stack coordi

            hr_h_end=ceil(h*rescale_factor);
            hr_h_start=hr_h_end-LF_cropped_size+1;

            hr_w_end=ceil(w*rescale_factor);
            hr_w_start=hr_w_end-LF_cropped_size+1;    %%HR coordi


            lr_h_end=ceil(hr_h_end/SR_factor);
            lr_h_start=lr_h_end-LR_cropped_size+1;
            lr_w_end=ceil(hr_w_end/SR_factor);
            lr_w_start=lr_w_end-LR_cropped_size+1;    %%LR coordi

            stack_region = gt_img(i : h, j : w,:);
            hr_region = hr_img(hr_h_start : hr_h_end, hr_w_start : hr_w_end,:);
            lr_region = lr_img(lr_h_start : lr_h_end, lr_w_start : lr_w_end,:);

            cropped_num_current=cropped_num_current+1;

            save_str=sprintf('Data%03d_idx%04d.tif',DatasetIndex,cropped_num_current);

            pixel_sum = sum(stack_region,'all');
            pixel_var = var(double(stack_region(:)));
            blk_max   = prctile(stack_region(:), 99.8);


            if save_all
                fprintf('%s:  sum %d var %d \n', save_str,pixel_sum, pixel_var );
                write3d(stack_region,fullfile(GT_SavePath,save_str),16);
                write3d(hr_region,fullfile(hr_SavePath,save_str),16);
                write3d(lr_region,fullfile(lr_SavePath,save_str),16);
                cropped_num=cropped_num+1;
            else
                %                 if blk_max >= local_thresh && pixel_var > var_threshold
                if (blk_max >= local_thresh)
                    fprintf('%s:  sum %d var %d pixel_ratio:%0.4f\n', save_str,pixel_sum, pixel_var,double(blk_max)/double(stack_p));
                    write3d(stack_region,fullfile(GT_SavePath,save_str),16);
                    write3d(hr_region,fullfile(hr_SavePath,save_str),16);
                    write3d(lr_region,fullfile(lr_SavePath,save_str),16);
                    cropped_num=cropped_num+1;
                end
            end
% 


        end
    end

    %     cropped_num=cropped_num_current+cropped_num;
end
disp(['cropped images : ' num2str(cropped_num)])

