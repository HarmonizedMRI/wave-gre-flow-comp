function pathValue = normalizeUserPath(pathValue)
    if isempty(pathValue)
        return;
    end
    if isstring(pathValue), pathValue = char(pathValue); end
    pathValue = strtrim(pathValue);
    if startsWith(pathValue, ['~' filesep]) || strcmp(pathValue, '~')
        homeDir = getenv('HOME');
        if isempty(homeDir) && ispc
            homeDir = getenv('USERPROFILE');
        end
        if strcmp(pathValue, '~')
            pathValue = homeDir;
        else
            pathValue = fullfile(homeDir, pathValue(3:end));
        end
    end
end
