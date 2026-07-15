function settings = configurePathSettings(scriptDir)
%CONFIGUREPATHSETTINGS Load, edit, validate, and save local path settings.
%
% The settings file is written beside the calling sequence script as:
%   gre_flash_path_settings.json
%
% Saved fields:
%   pulseq_path
%   safe_pns_prediction_path
%   out_path
%   system_asc_file
%
% When a saved file exists, the user is always asked whether to reuse it.
% If reused, the user may update one or more selected path entries. Pressing
% Enter for the output path uses MATLAB's current folder (pwd).

    if nargin < 1 || isempty(scriptDir)
        scriptDir = pwd;
    end
    scriptDir = normalizeUserPath(scriptDir);
    settingsFile = fullfile(scriptDir, 'gre_flash_path_settings.json');

    defaults = getWorkspaceDefaults();
    defaults.out_path = chooseNonempty(defaults.out_path, pwd);

    settings = defaults;
    useSaved = false;

    if exist(settingsFile, 'file')
        useSaved = askYesNo(sprintf('Use saved path settings from\n  %s?', settingsFile), true);
        if useSaved
            settings = loadSettingsFile(settingsFile, defaults);
            settings = validateAndRepairSettings(settings, defaults);

            if askYesNo('Change any saved path setting?', false)
                selectedFields = askWhichSettingsToChange();
                settings = promptSelectedSettings(settings, selectedFields);
                settings = validateAndRepairSettings(settings, defaults);
            end
        end
    end

    if ~useSaved
        fprintf('\nConfigure path settings. Existing workspace values are shown as defaults.\n');
        settings = promptSelectedSettings(defaults, fieldnamesInPromptOrder());
        settings = validateAndRepairSettings(settings, defaults);
    end

    settings = normalizeSettings(settings);
    saveSettingsFile(settingsFile, settings);
    assignSettingsToBaseWorkspace(settings);

    fprintf('Path settings saved to: %s\n', settingsFile);
    fprintf('Sequence output root: %s\n', settings.out_path);
end

function settings = getWorkspaceDefaults()
    settings = emptySettings();
    names = fieldnamesInPromptOrder();
    for ii = 1:numel(names)
        name = names{ii};
        if evalin('base', sprintf('exist(''%s'', ''var'')', name))
            value = evalin('base', name);
            if isstring(value), value = char(value); end
            if ischar(value)
                settings.(name) = normalizeUserPath(value);
            end
        end
    end
end

function settings = loadSettingsFile(settingsFile, defaults)
    settings = defaults;
    try
        rawText = fileread(settingsFile);
        loaded = jsondecode(rawText);
    catch ME
        warning('Could not read saved path settings: %s', ME.message);
        return;
    end

    names = fieldnamesInPromptOrder();
    for ii = 1:numel(names)
        name = names{ii};
        if isfield(loaded, name)
            value = loaded.(name);
            if isstring(value), value = char(value); end
            if ischar(value)
                settings.(name) = normalizeUserPath(value);
            end
        end
    end
end

function settings = validateAndRepairSettings(settings, defaults)
    settings = normalizeSettings(settings);

    if isempty(settings.pulseq_path) || ~exist(settings.pulseq_path, 'dir')
        if ~isempty(settings.pulseq_path)
            fprintf('Saved Pulseq path is invalid: %s\n', settings.pulseq_path);
        end
        settings = promptSelectedSettings(settings, {'pulseq_path'});
    end

    if ~isempty(settings.safe_pns_prediction_path) && ...
            ~exist(settings.safe_pns_prediction_path, 'dir')
        fprintf('Saved safe_pns_prediction path is invalid: %s\n', ...
            settings.safe_pns_prediction_path);
        settings = promptSelectedSettings(settings, {'safe_pns_prediction_path'});
    end

    if isempty(settings.out_path)
        settings.out_path = pwd;
    end
    if ~exist(settings.out_path, 'dir')
        createIt = askYesNo(sprintf('Output path does not exist:\n  %s\nCreate it?', ...
            settings.out_path), true);
        if createIt
            [ok, msg] = mkdir(settings.out_path);
            if ~ok
                error('Could not create output path %s: %s', settings.out_path, msg);
            end
        else
            settings.out_path = chooseNonempty(defaults.out_path, pwd);
            settings = promptSelectedSettings(settings, {'out_path'});
        end
    end

    if ~isempty(settings.system_asc_file) && ~exist(settings.system_asc_file, 'file')
        fprintf('Saved system ASC file is invalid: %s\n', settings.system_asc_file);
        settings = promptSelectedSettings(settings, {'system_asc_file'});
    end

    % Re-check required fields after any repairs.
    if isempty(settings.pulseq_path) || ~exist(settings.pulseq_path, 'dir')
        error('A valid pulseq_path is required.');
    end
    if isempty(settings.out_path) || ~exist(settings.out_path, 'dir')
        error('A valid out_path is required.');
    end
end

function settings = promptSelectedSettings(settings, selectedFields)
    for ii = 1:numel(selectedFields)
        fieldName = selectedFields{ii};
        currentValue = settings.(fieldName);

        switch fieldName
            case 'pulseq_path'
                settings.pulseq_path = promptDirectory( ...
                    'Pulseq path (repository root or matlab folder)', ...
                    currentValue, false, false, '');

            case 'safe_pns_prediction_path'
                settings.safe_pns_prediction_path = promptDirectory( ...
                    'safe_pns_prediction path (optional; enter - to clear)', ...
                    currentValue, true, false, '');

            case 'out_path'
                if ~isempty(currentValue)
                    fprintf('Current saved out_path: %s\n', currentValue);
                end
                settings.out_path = promptDirectory( ...
                    'Target output path for generated .seq files', ...
                    '', false, true, pwd);

            case 'system_asc_file'
                settings.system_asc_file = promptFile( ...
                    'System .asc file path (optional; enter - to clear)', ...
                    currentValue, true);

            otherwise
                error('Unknown path-setting field: %s', fieldName);
        end
    end
end

function selectedFields = askWhichSettingsToChange()
    names = fieldnamesInPromptOrder();
    labels = { ...
        'pulseq_path', ...
        'safe_pns_prediction_path', ...
        'out_path', ...
        'system_asc_file'};

    fprintf('\nWhich path setting(s) should be changed?\n');
    for ii = 1:numel(labels)
        fprintf('  %d) %s\n', ii, labels{ii});
    end
    fprintf('  5) all path settings\n');

    while true
        reply = strtrim(input('Enter one or more numbers separated by commas: ', 's'));
        tokens = regexp(reply, '[,\s]+', 'split');
        values = str2double(tokens);
        if isempty(reply) || any(isnan(values)) || any(values ~= round(values)) || ...
                any(values < 1) || any(values > 5)
            fprintf('Enter selections from 1 through 5.\n');
            continue;
        end
        if any(values == 5)
            selectedFields = names;
        else
            values = unique(values, 'stable');
            selectedFields = names(values);
        end
        return;
    end
end

function value = promptDirectory(promptText, currentValue, allowEmpty, createIfMissing, emptyDefault)
    if nargin < 5, emptyDefault = ''; end
    currentValue = normalizeUserPath(currentValue);

    while true
        % Print descriptive text separately from input(). MATLAB's Command
        % Window can wrap or misplace multi-line input prompts when a long
        % path is embedded directly in the prompt string.
        fprintf('\n%s\n', promptText);
        if ~isempty(currentValue)
            fprintf('  Current/default:\n');
            fprintf('    %s\n', currentValue);
            fprintf('  Action:\n');
            fprintf('    Enter a new path, or press Enter to keep the default.\n');
        elseif ~isempty(emptyDefault)
            fprintf('  Default:\n');
            fprintf('    %s\n', emptyDefault);
            fprintf('  Action:\n');
            fprintf('    Enter a path, or press Enter to use the default.\n');
        else
            fprintf('  Action:\n');
            fprintf('    Enter a directory path.\n');
        end
        reply = strtrim(input('  Path: ', 's'));

        if strcmp(reply, '-') && allowEmpty
            value = '';
            return;
        elseif isempty(reply)
            value = chooseNonempty(currentValue, emptyDefault);
            if isempty(value) && allowEmpty
                return;
            elseif isempty(value)
                fprintf('A valid directory path is required.\n');
                continue;
            end
        else
            value = normalizeUserPath(reply);
        end

        if exist(value, 'dir')
            return;
        end

        if createIfMissing && askYesNo(sprintf('Directory does not exist:\n  %s\nCreate it?', value), true)
            [ok, msg] = mkdir(value);
            if ok
                return;
            end
            fprintf('Could not create directory: %s\n', msg);
        else
            fprintf('Directory does not exist: %s\n', value);
        end
    end
end

function value = promptFile(promptText, currentValue, allowEmpty)
    currentValue = normalizeUserPath(currentValue);

    while true
        % Keep long file paths out of input() prompt strings so the Command
        % Window cursor remains aligned regardless of window width.
        fprintf('\n%s\n', promptText);
        if ~isempty(currentValue)
            fprintf('  Current/default:\n');
            fprintf('    %s\n', currentValue);
            fprintf('  Action:\n');
            fprintf('    Enter a new file path, or press Enter to keep the default.\n');
        else
            fprintf('  Action:\n');
            fprintf('    Enter a file path.\n');
        end
        reply = strtrim(input('  File: ', 's'));

        if strcmp(reply, '-') && allowEmpty
            value = '';
            return;
        elseif isempty(reply)
            value = currentValue;
            if isempty(value) && allowEmpty
                return;
            elseif isempty(value)
                fprintf('A valid file path is required.\n');
                continue;
            end
        else
            value = normalizeUserPath(reply);
        end

        if exist(value, 'file')
            return;
        end
        fprintf('File does not exist: %s\n', value);
    end
end

function tf = askYesNo(promptText, defaultValue)
    if defaultValue
        suffix = '[Y/n]';
    else
        suffix = '[y/N]';
    end

    while true
        % Separate multi-line question text from the short input prompt to
        % avoid Command Window alignment problems with long paths.
        fprintf('\n%s\n', promptText);
        reply = lower(strtrim(input(sprintf('  Choice %s: ', suffix), 's')));
        if isempty(reply)
            tf = logical(defaultValue);
            return;
        elseif any(strcmp(reply, {'y', 'yes', 'true', '1'}))
            tf = true;
            return;
        elseif any(strcmp(reply, {'n', 'no', 'false', '0'}))
            tf = false;
            return;
        end
        fprintf('Please answer yes or no.\n');
    end
end

function saveSettingsFile(settingsFile, settings)
    settingsToSave = struct;
    names = fieldnamesInPromptOrder();
    for ii = 1:numel(names)
        settingsToSave.(names{ii}) = settings.(names{ii});
    end

    try
        jsonText = jsonencode(settingsToSave);
        fid = fopen(settingsFile, 'wt');
        if fid < 0
            error('Could not open the settings file for writing.');
        end
        cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
        fprintf(fid, '%s\n', jsonText);
    catch ME
        warning('Could not save path settings JSON: %s', ME.message);
    end
end

function assignSettingsToBaseWorkspace(settings)
    names = fieldnamesInPromptOrder();
    for ii = 1:numel(names)
        assignin('base', names{ii}, settings.(names{ii}));
    end
end

function settings = normalizeSettings(settings)
    names = fieldnamesInPromptOrder();
    for ii = 1:numel(names)
        name = names{ii};
        if ~isfield(settings, name) || isempty(settings.(name))
            settings.(name) = '';
        else
            value = settings.(name);
            if isstring(value), value = char(value); end
            if ~ischar(value)
                value = '';
            end
            settings.(name) = normalizeUserPath(value);
        end
    end
end

function settings = emptySettings()
    settings = struct( ...
        'pulseq_path', '', ...
        'safe_pns_prediction_path', '', ...
        'out_path', '', ...
        'system_asc_file', '');
end

function names = fieldnamesInPromptOrder()
    names = {'pulseq_path', 'safe_pns_prediction_path', 'out_path', 'system_asc_file'};
end

function value = chooseNonempty(primaryValue, fallbackValue)
    if ~isempty(primaryValue)
        value = primaryValue;
    else
        value = fallbackValue;
    end
end
