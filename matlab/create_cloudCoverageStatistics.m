%% Create Cloudcover Statistics
% M.Brechbühler, GEO441

% define sensor and image type
sensor = 2; % S2 or L8
sfx='tif';

% choose sensor based on input
if sensor == 2
    DataDir = 'W:/Desktop/GEO441/data/FR_S2_original_filt/';
    cd(DataDir);
    files = dir(fullfile('.', ['FR_S2_*.' sfx])); % list available images
elseif sensor == 8
    DataDir = 'W:/Desktop/GEO441/data/FR_L8_original_filt/';
    cd(DataDir);
    files = dir(fullfile('.', ['FR_L8_*.' sfx])); % list available images
end

% set size of images
m = 3466;
n = 3666;

% set up vectors
date = zeros(1, length(files));
cc = zeros (1, length(files));

% add date and coverage percentage to vectors
for fn = 1:length(files)
    filename = files(fn).name;
    [im] = geotiffread(filename);
    date(fn) = convert2Date(filename);
    cc(fn) = (nnz(im(:,:,1))/(3466*3666))*100;
end
%}

% plot date and coverage
figure(1)
hold on
xlabel('Day, starting from 2015-01-01 [d]')
ylabel('Coverage percentage [%]')
title('S2 (original, filt) survey area coverage from 2015 to 2019')
plot(date, cc, 'x')
%plot(date, cc)
hold off

% save plot to file
print('FR_L8_original_filt_coverage','-dpng','-r300')

function day = convert2Date(filename)
    % converts a filename (Str) of the type
    % "FR_L8_30m_20150510_composite_filt" to a day of the year and adds
    % previous years
    tmp = split(filename, '_');
    tmpDate = insertAfter(insertAfter(tmp(4), 4,'-'), 7, '-');
    d = datevec(tmpDate);
    v = datenum(d);                % v now == [1957 12 25];
    day = v - datenum(d(1), 1,0)+(d(1)-2015)*365;  % datenum(yr,1,0) == datenum(yr-1,12,31)
end
