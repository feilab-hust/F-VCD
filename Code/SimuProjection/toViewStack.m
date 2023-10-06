clc
clear
addpath ./utils


view_num=3;
roi_size=[351,351];
img_size=1497;
pitch=499;


MLcenters=zeros(img_size,img_size);
p_center=ceil(pitch/2);
r=floor(pitch/2);

roi_radius= floor(roi_size/2);

d = pitch;
d1=0;
center=ceil(img_size/2);
fprintf('\n %d size --center %d',d,center)


MLcenters(center-ceil((d+d1)/2/sqrt(3)),center-ceil((d+d1)/2))=1;
MLcenters(center-ceil((d+d1)/2/sqrt(3)),center+ceil((d+d1)/2))=1;
MLcenters(center+ceil((d+d1)/sqrt(3)),center)=1;

[coordiy,coordix]=find(MLcenters==1);
coordi=zeros(length(coordiy),2);
for i=1:length(coordiy)
    coordi(i,1) =coordiy(i) ;
    coordi(i,2) =coordix(i) ;
end

[rank_co,ii]=sort(coordi(:,1));
temp=rank_co;
temp(:,2)=coordi(ii,2);
coordi=temp;

[hr_file_name,hr_filepath] = uigetfile('*.tif','Select HR Volumes','MultiSelect','on');
if ~iscell(hr_file_name)
    hr_file_name = {hr_file_name};
end
file_name=sprintf('\\SAI_499');
save_Path=fullfile(hr_filepath,file_name);
if exist(save_Path,'dir')==7
    ;
else
    mkdir(save_Path);
end
mask=@(xx,yy) xx.^2+yy.^2<=r^2;
[yy,xx]=meshgrid(1:pitch,1:pitch);
for img_idx=1:length(hr_file_name)
    img_name=hr_file_name{img_idx};
    fprintf('\n[%d/%d]preocessing %s',img_idx,length(hr_file_name),img_name);
    path=fullfile(hr_filepath,img_name);
    [img,bitDepth]=imread3d(path);
    ViewStack=zeros(pitch,pitch,view_num);
    
    for ii=1:length(coordi)
        temp_view=zeros(pitch,pitch);
        temp_slice=img(:,:,1);
        
        hy=coordi(ii,1);
        wx=coordi(ii,2);
        
 
        
        [iy,ix]=find(mask((yy-p_center),(xx-p_center))==1);
        for jj=1:length(iy)
            temp_view(iy(jj),ix(jj))=temp_slice(iy(jj)-p_center+hy,ix(jj)-p_center+wx);
        end
        ViewStack(:,:,ii)=temp_view;
    end
    save_path=fullfile(save_Path,img_name);
    ViewStack=ViewStack(p_center-roi_radius(1):p_center+roi_radius(1),p_center-roi_radius(2):p_center+roi_radius(2),:);
    write3d(ViewStack,save_path,bitDepth);
end









