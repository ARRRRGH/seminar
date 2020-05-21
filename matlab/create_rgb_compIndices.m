%% Create RGB images and videos from optical indices:
% M.Brechbühler, GEO441

% define your working directory
%DataDir = 'S:/course/geo441/data/2020_Camargue/';
DataDir = 'S:\course\geo441\stud\B_Camargue\indices\';
optical = [DataDir 'FR_L8S2_indices_composite_crop\'];
cd(optical);

% get an overview of the data present
sfx='tif';
files = dir(fullfile('.', ['L8_*.' sfx])); % list available images
% files = dir(fullfile('.', ['FR_S2_*.' sfx])); % list available images

%{
% open a single image at position i
i=2;
filename = files(i).name;
im = geotiffread(files(i).name);
im_info = geotiffinfo(files(i).name);
b_r = double(im(:,:,3));
b_r = b_r./max(b_r(:));

b_g = double(im(:,:,2));
b_g = b_g./max(b_g(:));

b_b = double(im(:,:,1));
b_b = b_b./max(b_b(:));

im3 = cat(3, b_r,  b_g, b_b);
im3 = imadjust(im3,stretchlim(im3),[]);
imagesc(im3);
%}

v = VideoWriter('W:/Desktop/geo441/output/L8_composite_indices_150dpi_2fps', 'MPEG-4');
v.FrameRate = 2; % Number of frames per second
open(v);

% set up figure
h = figure;
date = strings(1, length(files));

%for fn = 1:length(files)
for fn = 1:length(files)
    im = geotiffread(files(fn).name);
    im(im==-32767)=NaN;
    im_info = geotiffinfo(files(fn).name);
    %date(fn) = convert2Str(files(fn).name);
    
    % calculate bands
    b_r = double(im(:,:,3));
    b_r = b_r./max(b_r(:));
    b_g = double(im(:,:,1));
    b_g = b_g./max(b_g(:));
    b_b = double(im(:,:,2));
    b_b = b_b./max(b_b(:));
    im3 = cat(3, b_r,  b_g, b_b);
    im3 = imadjust(im3,stretchlim(im3),[]);
    
    %im1 = files(fn).name;
    %im2 = files2(fn).name;
    %[X1,map1]=imread(im1);
    %[X2,map2]=imread(mndwi);
    %subplot(1,2,1), imshow(X1,map1)
    %subplot(1,2,2), imshow(X2,map2)
    % Display image name in the command window.
    fprintf(1, '%d Now reading %s\n', fn, files(fn).name);
    
    % overlay filename
    position = [50 50]; 
    box_color = 'yellow';
    im3_overlay = insertText(im3,position,files(fn).name,'FontSize',50,'BoxColor',box_color,'BoxOpacity',0.4,'TextColor','white');
    imshow(im3_overlay);
    
    print('W:/Desktop/geo441/output/plot_tmp','-dpng','-r150');
    Frame = imread('W:/Desktop/geo441/output/plot_tmp.png');
    writeVideo(v, Frame);
end

close(v);

function day = convert2Str(filename)
    % converts a filename (Str) of the type
    % "FR_L8_30m_20150510_composite_filt" to a str date
    tmp = split(filename, '_');
    tmpDate = insertAfter(insertAfter(tmp(4), 4,'-'), 7, '-');
    d = datevec(tmpDate);
    v = datenum(d);                % v now == [1957 12 25];
    day = datestr(v, 'dd-mm-yy');
end
