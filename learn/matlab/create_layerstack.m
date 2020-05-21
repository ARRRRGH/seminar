%% Create a layerstack from SAR and optical images
% M.Brechbühler, GEO441

% define your working directory
%DataDir = 'S:/course/geo441/data/2020_Camargue/';
DataDir = 'W:/Desktop/GEO441/output/test_yearly classification/';
cd(DataDir);
coordRefSysCode = 32631;

% Multiple files
% get an overview of the data present
sfx='tif';
files_opt = dir(fullfile('.', ['FR_L8*.' sfx])); % list available images
files_sar = dir(fullfile('.', ['S1*.' sfx])); % list available images
for y =2016:2016 
    for m = 1:12
        filename = files_opt(1).name;
        [Aopt,Ropt] = geotiffread(filename);
        filename = files_sar(1).name;
        [Asar,Rsar] = geotiffread(filename);
        b1 = im(:,:,1);
        b2 = im(:,:,2);
        b3 = im(:,:,3);
        b4 = im(:,:,4);
        b5 = im(:,:,5);
        b6 = im(:,:,6);
        b7 = im(:,:,7);
        b8 = im(:,:,8);
        b9 = im(:,:,9);
        b_sar = single(rescale(Asar(:,:,1),0,10000));
        b_sar(b1==0)=0;
        fV = cat(3,b_sar,b1,b2,b3,b4,b5,b6,b7,b8,b9);
        geotiffwrite([DataDir, 'layerstack_', int2str(y), '_', int2str(m), '.tif'], fV, Ropt, 'CoordRefSysCode', coordRefSysCode);
    end
end