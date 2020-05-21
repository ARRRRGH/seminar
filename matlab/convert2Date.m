%% Convert filename to date
% M.Brechbühler, GEO441

function day = convert2Date(filename)
    % converts a filename (Str) of the type "FR_L8_30m_20150510_composite_filt" to a day of the year
    tmp = split(filename, '_');
    tmpDate = insertAfter(insertAfter(tmp(4), 4,'-'), 7, '-');
    d = datevec(tmpDate);
    v = datenum(d);                % v now == [1957 12 25];
    day = v - datenum(d(1), 1,0);  % datenum(yr,1,0) == datenum(yr-1,12,31)
end
