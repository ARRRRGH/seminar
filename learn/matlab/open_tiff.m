%% Open and read .geotiff file:
% M.Brechbühler, GEO441

% define your working directory
%DataDir = 'S:/course/geo441/data/2020_Camargue/';
DataDir = 'W:/Desktop/GEO441\data\FR_L8_composite_filt_crop';
cd(DataDir);

%{
% Single file
filename = 'FR_L8_30m_20150102_composite_filt.tif';
[A,R] = geotiffread(filename);
imshow(A)
%imwrite(A,[filename '.jpg'],'JPEG');
%}

% Single file (RGB)
filename = 'FR_L8_30m_20181124_composite_filt_crop.tif';
[im,im_info] = geotiffread(filename);
%b_r = double(im(:,:,4));
%b_r = b_r./max(b_r(:));

%b_g = double(im(:,:,3));
%b_g = b_g./max(b_g(:));

%b_b = double(im(:,:,2));
%b_b = b_b./max(b_b(:));

%im3 = cat(3, b_r,  b_g, b_b);
%im3 = imadjust(im3,stretchlim(im3),[]);
%imagesc(im3);


%{
% Multiple files
% get an overview of the data present
sfx='tif';
files = dir(fullfile('.', ['test_*.' sfx])); % list available images

for fn = 1:length(files)
    filename = files(fn).name;
    [A,R] = geotiffread(filename);
    %imwrite(A,[filename '.jpg'],'JPEG');
end
%}