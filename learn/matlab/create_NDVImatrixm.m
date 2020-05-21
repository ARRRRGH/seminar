%% Create NDVI matrix:
% M.Brechbühler, GEO441

% define your working directory
%DataDir = 'S:/course/geo441/data/2020_Camargue/';
sensor = 8; % S2 or L8
sfx='tif';

if sensor == 2
    DataDir = 'W:/Desktop/GEO441/data/FR_S2_composite_filt/';
    cd(DataDir);
    files = dir(fullfile('.', ['FR_S2_*.' sfx])); % list available images
elseif sensor == 8
    DataDir = 'W:/Desktop/GEO441/data/FR_L8_composite_filt/';
    cd(DataDir);
    files = dir(fullfile('.', ['FR_L8_*.' sfx])); % list available images
end

% set size of images
%m = 3466;
%n = 3666;

%ndviData = zeros(m, n, length(files));
date = zeros(1, length(files));

for fn = 1:length(files)
    filename = files(fn).name;
    [A,R] = geotiffread(filename);
    date(fn) = convert2Date(filename);
    if sensor == 2
        NDVI = double(A(:,:,7)-A(:,:,3)) ./ double(A(:,:,7)+A(:,:,3));
    elseif sensor == 8
        NDVI = double(A(:,:,5)-A(:,:,3)) ./ double(A(:,:,5)+A(:,:,3));
    end
    %ndviData(:,:,fn) = NDVI;
    imwrite(NDVI, ['NDVI/' filename '_NDVI.tif'],'Mode','lossless');
end

function day = convert2Date(filename)
    % converts a filename (Str) of the type "FR_L8_30m_20150510_composite_filt" to a day of the year
    tmp = split(filename, '_');
    tmpDate = insertAfter(insertAfter(tmp(4), 4,'-'), 7, '-');
    d = datevec(tmpDate);
    v = datenum(d);                % v now == [1957 12 25];
    day = v - datenum(d(1), 1,0);  % datenum(yr,1,0) == datenum(yr-1,12,31)
end