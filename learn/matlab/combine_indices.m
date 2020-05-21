%% Combine L8 and S2 derived NDVI, NDWI and NDBI to get more coverage
% M.Brechbühler, GEO441

% set reference system for GEOTIFF exports
coordRefSysCode = 32631;
x_size = 1897;
y_size = 1497;
noData = -32767;

%% set file directories
DataDir = 'S:\course\geo441\stud\B_Camargue\';
OutputDir = 'S:\course\geo441\stud\B_Camargue\indices\yearly\';
cd(DataDir);
l8_ndvi_path = '\indices\FR_L8_composite_filt_NDVI_crop\';
s2_ndvi_path = '\indices\FR_S2_composite_filt_NDVI_crop\';
l8_ndwi_path = '\indices\FR_L8_composite_filt_NDWI_crop\';
s2_ndwi_path = '\indices\FR_S2_composite_filt_NDWI_crop\';
l8_ndbi_path = '\indices\FR_L8_composite_filt_NDBI_crop\';
s2_ndbi_path = '\indices\FR_S2_composite_filt_NDBI_crop\';

%% load files
sfx='tif';
l8_ndvi = dir(fullfile('.', [l8_ndvi_path, 'FR_*.' sfx])); % list available images
l8_ndwi = dir(fullfile('.', [l8_ndwi_path, 'FR_*.' sfx]));
l8_ndbi = dir(fullfile('.', [l8_ndbi_path, 'FR_*.' sfx]));
s2_ndvi = dir(fullfile('.', [s2_ndvi_path, 'FR_*.' sfx]));
s2_ndwi = dir(fullfile('.', [s2_ndwi_path, 'FR_*.' sfx]));
s2_ndbi = dir(fullfile('.', [s2_ndbi_path, 'FR_*.' sfx]));

%% add date to filestructs
for fn=1:length(l8_ndvi)
    l8_ndvi(fn).date = convert2date(getfield(l8_ndvi(fn), 'name'));
    l8_ndwi(fn).date = convert2date(getfield(l8_ndwi(fn), 'name'));
    l8_ndbi(fn).date = convert2date(getfield(l8_ndbi(fn), 'name'));
end

for fn=1:length(s2_ndvi)
    s2_ndvi(fn).date = convert2date(getfield(s2_ndvi(fn), 'name'));
    s2_ndwi(fn).date = convert2date(getfield(s2_ndwi(fn), 'name'));
    s2_ndbi(fn).date = convert2date(getfield(s2_ndbi(fn), 'name'));
end

%% combine images monthly

for y=2015:2018
    for m=1:12
        % creating NDVI monthly composites
        fprintf(1, '(%d-%d) Now creating NDVI\n', y, m);
        NDVI = zeros(y_size, x_size);
        NDVI = NDVI+(NDVI==0).*noData;
        %{
        for fn=1:length(l8_ndvi)
            if l8_ndvi(fn).date == [y,m]
                [A,R] = geotiffread([DataDir, l8_ndvi_path, l8_ndvi(fn).name]);
                NDVI(NDVI==noData)=A(NDVI==noData);
            end
        end
        %}
        for fn=1:length(s2_ndvi)
            if s2_ndvi(fn).date == [y,m]
                [A,R] = geotiffread([DataDir, s2_ndvi_path, s2_ndvi(fn).name]);
                NDVI(NDVI==noData)=A(NDVI==noData);
            end
        end
        NDVI_int16 = int16(NDVI);
        %geotiffwrite([OutputDir, '/NDVI/' int2str(y), '_', int2str(m), '_NDVI.tif'], NDVI_int16, R, 'CoordRefSysCode', coordRefSysCode);
        
        % creating NDWI monthly composites
        fprintf(1, '(%d-%d) Now creating NDWI\n', y, m);
        NDWI = zeros(y_size, x_size);
        NDWI = NDWI+(NDWI==0).*noData;
        %{
        for fn=1:length(l8_ndwi)
            if l8_ndwi(fn).date == [y,m]
                [A,R] = geotiffread([DataDir, l8_ndwi_path, l8_ndwi(fn).name]);
                NDWI(NDWI==noData)=A(NDWI==noData);
            end
        end
        %}
        for fn=1:length(s2_ndwi)
            if s2_ndwi(fn).date == [y,m]
                [A,R] = geotiffread([DataDir, s2_ndwi_path, s2_ndwi(fn).name]);
                NDWI(NDWI==noData)=A(NDWI==noData);
            end
        end
        NDWI_int16 = int16(NDWI);
        %geotiffwrite([OutputDir, '/NDWI/' int2str(y), '_', int2str(m), '_NDWI.tif'], NDWI_int16, R, 'CoordRefSysCode', coordRefSysCode);
        
        % creating NDBI monthly composites
        fprintf(1, '(%d-%d) Now creating NDBI\n', y, m);
        NDBI = zeros(y_size, x_size);
        NDBI = NDBI+(NDBI==0).*noData;
        %{
        for fn=1:length(l8_ndbi)
            if l8_ndbi(fn).date == [y,m]
                [A,R] = geotiffread([DataDir, l8_ndbi_path, l8_ndbi(fn).name]);
                NDBI(NDBI==noData)=A(NDBI==noData);
            end
        end
        %}
        for fn=1:length(s2_ndbi)
            if s2_ndbi(fn).date == [y,m]
                [A,R] = geotiffread([DataDir, s2_ndbi_path, s2_ndbi(fn).name]);
                NDBI(NDBI==noData)=A(NDBI==noData);
            end
        end
        NDBI_int16 = int16(NDBI);
        %geotiffwrite([OutputDir, '/NDBI/', int2str(y), '_', int2str(m), '_NDBI.tif'], NDBI_int16, R, 'CoordRefSysCode', coordRefSysCode);
        
        % combine indices
        indices = cat(3,NDVI,NDWI,NDBI);
        indices_int16 = int16(indices);
        geotiffwrite([OutputDir, 'S2_', int2str(y), '_', int2str(m), '_indices.tif'], indices_int16, R, 'CoordRefSysCode', coordRefSysCode);
        %geotiffwrite([OutputDir, 'L8_', int2str(y), '_', int2str(m), '_indices.tif'], indices_int16, R, 'CoordRefSysCode', coordRefSysCode);
    end
end

function date = convert2date(filename)
    % converts a filename (Str) of the type "FR_L8_30m_20150510_composite_filt" to a day of the year
    tmp = split(filename, '_');
    tmpDate = insertAfter(insertAfter(tmp(4), 4,'-'), 7, '-');
    d = datevec(tmpDate);
    date = d(1:2);
end