%% Create water abundance and minimal/maximal extent maps from optical indices:
% M.Brechbühler, GEO441

% define your working directory
DataDir = 'S:\course\geo441\stud\B_Camargue\indices\FR_L8_composite_filt_mNDWI_crop\';
sfx='tif';

cd(DataDir);
files = dir(fullfile('.', ['FR_L8_*.' sfx])); % list available images

% set size of images
m = 1497;
n = 1897;

%set reference system for GEOTIFF exports
coordRefSysCode = 32631;

%init abundance map
count = zeros(m, n);
flag = zeros(m, n);
nanval = -32767;

%init min/max map
minExt = ones(m, n);
maxExt = zeros(m, n);

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDWI', fn, files(fn).name);
    filename = files(fn).name;
    [A,R] = geotiffread(filename);
    watermask = A>3870;
    count = count + double(watermask); %Threshold from Acharya et al. 2018
    flag = (flag + (A==nanval));
    maxExt(watermask==1)=1;
    minExt(watermask==0)=0;
end
% calculate percentage
occurence = ones(m, n)*length(files);
occurence = occurence - flag;
percentage = double(count./occurence)*100;

% rescale to get 100% over permanent water (e.g. open sea)
%percentage(percentage >= 89.66) = 89.66;
%scaled_percentage = rescale(percentage,0,100);
%show image

imagesc(percentage)
colormap(pink)
colorbar
caxis([0 100])
%xlim([500 2800])
%ylim([2000 3450])
set(gca,'DataAspectRatio',[1 1 1])
title({'Camargue, Water Occurence (2015-2019) [%]'; 'mNDWI tresholding using L8 composites'})
%%
% write files
geotiffwrite('L8_watermask_abundance.tif', percentage, R, 'CoordRefSysCode', coordRefSysCode);
%geotiffwrite('L8_watermask_max.tif', maxExt==1, R, 'CoordRefSysCode', coordRefSysCode);
%geotiffwrite('L8_watermask_min.tif', minExt==1, R, 'CoordRefSysCode', coordRefSysCode);