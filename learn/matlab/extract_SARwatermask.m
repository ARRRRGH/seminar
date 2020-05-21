% Waterbody Extraction from SAR imagery with Thresholding
% M.Brechbühler, GEO441 - Remote Sensing Seminar, 29.04.2020

% years: 2015 (1-53), 2016 (54-113), 2017 (114-174), 2018 (175-235), 
% 2019 (236-244)
%% Set file directories

DataDir = 'W:\Desktop\GEO441\m_code\';
input = [DataDir 'data/'];
cd(input);

sfx='tif';
files = dir(fullfile('.', ['S1IWGRDH*.' sfx])); % list available images

%choose mode
thresh_mode = 'manual'; %'otsu_valley_emphasis'; %manual, otsu, otsu_valley_emphasis

%init matrices for min/max extent and abundance map
maxExt = zeros(1497,1897);
minExt = ones(1497,1897);
count = zeros(1497,1897);
maxCount = 0;

%set reference system for GEOTIFF exports
coordRefSysCode = 32631;

%% Thresholding of bimodal SAR backscatter distribution

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'SAR', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    if range(A(:))>100
        fprintf(1, '\tImage range unusual, no thresholding possible!\n');
    else 
        maxCount = maxCount+1;
        im = mat2gray(A);
        thresh_valley_emphasis_grayscale = valley_emphasis(im);
        thresh_valley_emphasis = range(A(:))/265*thresh_valley_emphasis_grayscale+min(A(:));
        thresh_otsu = multithresh(A(:));
        % Choose threshhold based on mode and create watermask
        if strcmp(thresh_mode,'manual')
            %plot histogram and Otsu/VE-Otsu thresholds
            figure(1) 
            histogram(A)
            xlabel(['Backscatter Intensity [dB]']);
            ylabel(['Pixel Count']);
            hold on
            line([thresh_valley_emphasis, thresh_valley_emphasis], ylim, 'LineWidth', 2, 'Color', 'r');
            line([thresh_otsu, thresh_otsu], ylim, 'LineWidth', 2, 'Color', 'b');
            hold off
            [x,y] = ginput(1);
            thresh = x;
        elseif strcmp(thresh_mode,'otsu')
            thresh = thresh_otsu;
        elseif strcmp(thresh_mode,'otsu_valley_emphasis')
            thresh = thresh_valley_emphasis;
        end
        % create watermask
        watermask = A<thresh;
        % update max,min and abundance map for plot
        count = count+watermask;
        maxExt(watermask==1)=1;
        minExt(watermask==0)=0;
        % print threshold
        fprintf(1, '\tThreshold set at %d\n', thresh);
        % export single mask in .geotiff
        %geotiffwrite(['output/', erase(files(fn).name, '.tif'), '_watermask.tif'], watermask, R, 'CoordRefSysCode', coordRefSysCode);
    end
end

%% Write to imagefile
%geotiffwrite('max_watermask.tif', maxExt==1, R, 'CoordRefSysCode', coordRefSysCode);
%geotiffwrite('min_watermask.tif', minExt==1, R, 'CoordRefSysCode', coordRefSysCode);
%abundance_watermask = (count./maxCount).*100;
%geotiffwrite('abundance_watermask.tif', (count./maxCount).*100, R, 'CoordRefSysCode', coordRefSysCode);

%% Plot water abundance map from watermask
%{
%plot abundance map
imagesc(abundance_watermask)
colormap(pink)
colorbar
caxis([0 100])
%xlim([500 2800])
%ylim([2000 3450])
set(gca,'DataAspectRatio',[1 1 1])
%}

%% Revised Valley-emphasis Otsu
% Copyright (C) 2017, Mai Thanh Nhat Truong, All rights reserved.
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3,
% as published by the Free Software Foundation.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% An implementation of
% H. F. Ng. Automatic thresholding for defect detection.
% Pattern Recognition Letters, 27(14):1644-1649, 2006.

function output = valley_emphasis(I)
% Input:
%   I : 8-bit gray scale image
% Output:
%   output : optimal threshold value,
%            binary image can be obtained by using `im2bw(I, output/255)`

    [COUNTS,X] = imhist(I);

    % Total number of pixels
    total = size(I,1)*size(I,2);

    sumVal = 0;
    for t = 1 : 256
        sumVal = sumVal + ((t-1) * COUNTS(t)/total);
    end

    varMax = 0;
    threshold = 0;

    omega_1 = 0;
    omega_2 = 0;
    mu_1 = 0;
    mu_2 = 0;
    mu_k = 0;

    for t = 1 : 256
        omega_1 = omega_1 + COUNTS(t)/total;
        omega_2 = 1 - omega_1;
        mu_k = mu_k + (t-1) * (COUNTS(t)/total);
        mu_1 = mu_k / omega_1;
        mu_2 = (sumVal - mu_k)/(omega_2);
        currentVar = (1 - COUNTS(t)/total) * (omega_1 * mu_1^2 + omega_2 * mu_2^2);
        % Check if new maximum found
        if (currentVar > varMax)
           varMax = currentVar;
           threshold = t-1;
        end
    end

    output = threshold;
end