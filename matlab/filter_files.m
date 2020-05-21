%% Filters the files for cloud coverage threshold
% M.Brechbühler, GEO441

% define your working directory
%DataDir = 'S:/course/geo441/data/2020_Camargue/';
DataDir = 'W:/Desktop/GEO441/output/classification_test/';
cd(DataDir);
coordRefSysCode = 32631;

% Multiple files
% get an overview of the data present
sfx='tif';
files_opt = dir(fullfile('.', ['FR_L8*.' sfx])); % list available images

for fn=1:length(files_opt)
    filename = files_opt(fn).name;
    [Aopt,Ropt] = geotiffread(filename);
    coverage = nnz(Aopt)/numel(Aopt)*100;
    if coverage < 75
        delete(files_opt(fn).name);
    end
    %files_opt(fn).coverage = coverage;
end
