%% Create optical Indices for Landsat 8:
% M.Brechbühler, GEO441

% Original Bands:   1   2   3   4   5   6   7  T1  T2   (L8)
% Input bands:      1   2   3   4   5   6   7   8   9   (Data)
% Band descr.: cb, blue, green, red, nir, swir1, swir2, tir1, tir2
% Index-Values from [-1, 1] saved as int16 from [0, 10000] (factor 10'000)
% NaN-Values (cloud cover) are set to -32767
%% set file directories
DataDir = 'W:/Desktop/';
optical = [DataDir 'GEO441/data/FR_L8_composite_filt_crop/'];
cd(optical);

sfx='tif';
files = dir(fullfile('.', ['FR_L8_*.' sfx])); % list available images

% set reference system for GEOTIFF exports
coordRefSysCode = 32631;

%% calculate and save NDVI
% NDVI: (NIR-Red)/(NIR+Red)
% (band5-band4)/(band5+band4) (L8)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDVI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDVI = (double(A(:,:,5))-double(A(:,:,4))) ./ (double(A(:,:,5))+double(A(:,:,4)));
    NDVI = int16(NDVI*10000);
    NDVI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDVI.tif'], NDVI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save NDMI
% NDMI: (NIR-SWIR)/(NIR+SWIR)
% (band5-band6)/(band5+band6) (L8)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDMI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDMI = (double(A(:,:,5))-double(A(:,:,6))) ./ (double(A(:,:,5))+double(A(:,:,6)));
    NDMI = int16(NDMI*10000);
    NDMI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDMI.tif'], NDMI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save mNDWI
% mNDWI: (Green-SWIR)/(Green+SWIR)
% (band3-band6)/(band3+band6) (L8)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'mNDWI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    mNDWI = (double(A(:,:,3))-double(A(:,:,6))) ./ (double(A(:,:,3))+double(A(:,:,6)));
    mNDWI = int16(mNDWI*10000);
    mNDWI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_mNDWI.tif'], mNDWI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save NDBI
% NDBI: (SWIR-NIR)/(SWIR+NIR)
% (band6-band5)/(band6+band5) (L8) 

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDBI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDBI = (double(A(:,:,6))-double(A(:,:,5))) ./ (double(A(:,:,6))+double(A(:,:,5)));
    NDBI = int16(NDBI*10000);
    NDBI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDBI.tif'], NDBI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save EVI
% EVI
% 2.5 * ((Band 5 – Band 4) / (Band 5 + 6 * Band 4 – 7.5 * Band 2 + 1)) (L8)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'EVI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    EVI = 2.5*((double(A(:,:,5))-double(A(:,:,4))) ./ (double(A(:,:,5))+6*double(A(:,:,4))-7.5*double(A(:,:,2))+1));
    EVI = int16(EVI*10000);
    EVI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_EVI.tif'], EVI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save NDSI (Sainity)
% NDSI: (SWIR1-SWIR2)/(SWIR1+SWIR2)
% (band6-band7)/(band6+band7) (L8) 

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDSI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDSI = (double(A(:,:,6))-double(A(:,:,7))) ./ (double(A(:,:,6))+double(A(:,:,7)));
    NDSI = int16(NDSI*10000);
    NDSI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDSI.tif'], NDSI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save NDBaI
% NDBaI: (SWIR1-TIR1)/(SWIR1+TIR1)
% (band6-band8)/(band6+band8) (L8) 

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDBaI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDBaI = (double(A(:,:,6))-double(A(:,:,8))) ./ (double(A(:,:,6))+double(A(:,:,8)));
    NDBaI = int16(NDBaI*10000);
    NDBaI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDBaI.tif'], NDBaI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save UI
% UI: (SWIR2-NIR)/(SWIR2+NIR)
% (band7-band5)/(band7+band5) (L8) 

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'UI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    UI = (double(A(:,:,7))-double(A(:,:,5))) ./ (double(A(:,:,7))+double(A(:,:,5)));
    UI = int16(UI*10000);
    UI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_UI.tif'], UI, R, 'CoordRefSysCode', coordRefSysCode);
end
