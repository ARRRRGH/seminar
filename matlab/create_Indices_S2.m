%% Create optical Indices for Sentinel-2:
% M.Brechbühler, GEO441

% Original Bands:   2   3   4   5   6   7   8   8a  11  12  (S2)
% Input bands:      1   2   3   4   5   6   7   8   9   10  (Data)
% Band descr.: blue, green, red, re1, re2, re3, nir1, nir2, swir1, swir2 (Data)
% Index-Values from [-1, 1] saved as uint16 from [0, 10000] (factor 10'000)
% NaN-Values (cloud cover) are set to -32767

%% set file directories
DataDir = 'W:/Desktop/';
optical = [DataDir 'GEO441/data/FR_S2_composite_filt_crop/'];
cd(optical);

sfx='tif';
files = dir(fullfile('.', ['FR_S2_*.' sfx])); % list available images

% set reference system for GEOTIFF exports
coordRefSysCode = 32631;

%% calculate and save NDVI
% NDVI: (NIR-Red)/(NIR+Red)
% (band8-band4)/(band8+band4) (S2)
% (band7-band3)/(band7+band3) (Data)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDVI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDVI = (double(A(:,:,7))-double(A(:,:,3))) ./ (double(A(:,:,7))+double(A(:,:,3)));
    NDVI = int16(NDVI*10000);
    NDVI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDVI.tif'], NDVI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save NDMI
% NDMI: (NIR-SWIR)/(NIR+SWIR)
% (band8-band11)/(band8+band11) (S2)
% (band7-band9)/(band7+band9) (Data)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDMI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDMI = (double(A(:,:,7))-double(A(:,:,9))) ./ (double(A(:,:,7))+double(A(:,:,9)));
    NDMI = int16(NDMI*10000);
    NDMI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDMI.tif'], NDMI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save mNDWI
% mNDWI: (Green-SWIR)/(Green+SWIR)
% (band3-band11)/(band3+band11) (S2)
% (band2-band9)/(band2+band9) (Data)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'mNDWI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    mNDWI = (double(A(:,:,2))-double(A(:,:,9))) ./ (double(A(:,:,2))+double(A(:,:,9)));
    mNDWI = int16(mNDWI*10000);
    mNDWI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_mNDWI.tif'], mNDWI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save NDBI
% NDBI: (SWIR-NIR)/(SWIR+NIR)
% (band11-band8)/(band11+band8) (S2)
% (band9-band7)/(band9+band7) (Data)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDBI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDBI = (double(A(:,:,9))-double(A(:,:,7))) ./ (double(A(:,:,9))+double(A(:,:,7)));
    NDBI = int16(NDBI*10000);
    NDBI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDBI.tif'], NDBI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save EVI
% EVI
% 2.5 * ((Band 8 – Band 4) / (Band 8 + 6 * Band 4 – 7.5 * Band 2 + 1)) (S2)
% 2.5 * ((Band 7 – Band 3) / (Band 7 + 6 * Band 3 – 7.5 * Band 1 + 1)) (Data)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'EVI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    EVI = 2.5*((double(A(:,:,7))-double(A(:,:,3))) ./ (double(A(:,:,7))+6*double(A(:,:,3))-7.5*double(A(:,:,1))+1));
    EVI = int16(EVI*10000);
    EVI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_EVI.tif'], EVI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save NDSI (Salinity)
% NDSI: (SWIR1-SWIR2)/(SWIR1+SWIR2)
% (band11-band12)/(band11+band12) (S2)
% (band9-band10)/(band9+band10) (Data)

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'NDSI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    NDSI = (double(A(:,:,9))-double(A(:,:,10))) ./ (double(A(:,:,9))+double(A(:,:,10)));
    NDSI = int16(NDSI*10000);
    NDSI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_NDSI.tif'], NDSI, R, 'CoordRefSysCode', coordRefSysCode);
end

%% calculate and save UI
% UI: (SWIR2-NIR)/(SWIR2+NIR)
% (band12-band8)/(band12+band8) (S2)
% (band10-band7)/(band10+band7) (Data) 

for fn = 1:length(files)
    fprintf(1, '(%s-%d) Now reading %s\n', 'UI', fn, files(fn).name);
    [A,R] = geotiffread(files(fn).name);
    UI = (double(A(:,:,10))-double(A(:,:,7))) ./ (double(A(:,:,10))+double(A(:,:,7)));
    UI = int16(UI*10000);
    UI(A(:,:,4)==0)=-32767;
    geotiffwrite([erase(files(fn).name, '.tif'), '_UI.tif'], UI, R, 'CoordRefSysCode', coordRefSysCode);
end
