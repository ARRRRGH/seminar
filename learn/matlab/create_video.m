%% Create RGB video from scenes:
% M.Brechbühler, GEO441

v = VideoWriter('L8_original_filt_150dpi_8fps', 'MPEG-4');
v.FrameRate = 2; % Number of frames per second
open(v);

% set up figure
h = figure;

% define your working directory
DataDir = 'S:/course/geo441/stud/B_Camargue/indices/';
%DataDir = 'W:/Desktop/data/';
optical = [DataDir 'yearly/'];
cd(optical);

% get an overview of the data present
sfx='.tif';
files = dir(fullfile('.', ['S2_*' sfx])); % list available images
%mndwiFiles = dir(fullfile('.', ['FR_L78S2_*_model_BandMath.tif.' sfx])); % list available images

for fn = 1:length(files)
    fprintf(1, '(%d) Now reading %s\n', fn, files(fn).name);
    im = geotiffread(files(fn).name);
    im_info = geotiffinfo(files(fn).name);
    b_r = double(im(:,:,3));
    b_r = b_r./max(b_r(:));
    b_g = double(im(:,:,1));
    b_g = b_g./max(b_g(:));
    b_b = double(im(:,:,2));
    b_b = b_b./max(b_b(:));
    im3 = cat(3, b_r,  b_g, b_b);
    im3 = imadjust(im3,stretchlim(im3),[]);
    imagesc(im3);
    currentFigure = gcf;
    title(currentFigure.Children(end), files(fn).name);
    %xlim([0 375])
    %ylim([0 250])
    %caxis([0 25])
    print('plot_tmp','-dpng','-r150')
    Frame = imread('plot_tmp.png');
    writeVideo(v, Frame);
end

close(v);

function date = convertDate(filename)
    % converts a filename (Str) of the type "FR_L8_30m_20150510_composite_filt" to a day of the year
    tmp = split(filename, '_');
    tmpDate = insertAfter(insertAfter(tmp(4), 4,'-'), 7, '-');
    d = datevec(tmpDate);
    v = datenum(d);                % v now == [1957 12 25];
    date = datestr(v);
end