function pathValue = ensureTrailingFilesep(pathValue)
    if isempty(pathValue)
        return;
    end
    if pathValue(end) ~= filesep
        pathValue = [pathValue filesep];
    end
end
