%% Create layerstack from Optical Indices and SAR:
% M.Brechbühler, GEO441

% define your working directory
%DataDir = 'S:/course/geo441/data/2020_Camargue/';
DataDir = 'W:/Desktop/GEO441/output\2016_feb_stack/';
cd(DataDir);
coordRefSysCode = 32631;

% Multiple files
% get an overview of the data present
sfx='tif';
files_opt = dir(fullfile('.', ['FR*.' sfx])); % list available images
files_sar = dir(fullfile('.', ['S1*.' sfx])); % list available images
filename = files_opt(1).name;
[Aopt,Ropt] = geotiffread(filename);
filename = files_sar(1).name;
[Asar,Rsar] = geotiffread(filename);
filename = files_sar(2).name;
[Asar2,Rsar2] = geotiffread(filename);
b1 = Aopt(:,:,1);
b2 = Aopt(:,:,2);
b3 = Aopt(:,:,3);
b4 = Aopt(:,:,4);
b5 = Aopt(:,:,5);
b6 = Aopt(:,:,6);
b7 = Aopt(:,:,7);
b8 = Aopt(:,:,8);
b9 = Aopt(:,:,9);
b10 = Aopt(:,:,10);
b_sar1 = single(rescale(Asar(:,:,1),0,10000));
b_sar1(b1==0)=0;
b_sar2 = single(rescale(Asar2(:,:,1),0,10000));
b_sar2(b1==0)=0;
fV = cat(3,b_sar1,b_sar2,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10);
geotiffwrite([DataDir, 'layerstack.tif'], fV, Ropt, 'CoordRefSysCode', coordRefSysCode);
