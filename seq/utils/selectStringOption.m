function value = selectStringOption(varName, promptText, options, defaultValue)
    if nargin < 4 || isempty(defaultValue)
        defaultValue = options{1};
    end

    value = defaultValue;
    if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        workspaceValue = evalin('base', varName);
        if isstring(workspaceValue), workspaceValue = char(workspaceValue); end
        if ischar(workspaceValue) && any(strcmp(workspaceValue, options))
            value = workspaceValue;
        else
            warning('Ignoring invalid workspace value for %s.', varName);
        end
    end

    defaultIdx = find(strcmp(value, options), 1);
    if isempty(defaultIdx), defaultIdx = 1; end

    fprintf('\n%s:\n', promptText);
    for ii = 1:numel(options)
        fprintf('  %2d) %s\n', ii, options{ii});
    end
    reply = input(sprintf('Select 1-%d [default %d: %s]: ', numel(options), defaultIdx, options{defaultIdx}), 's');

    if ~isempty(strtrim(reply))
        numericChoice = str2double(reply);
        if ~isnan(numericChoice) && numericChoice == round(numericChoice) && numericChoice >= 1 && numericChoice <= numel(options)
            value = options{numericChoice};
        else
            trimmedReply = strtrim(reply);
            idx = find(strcmp(trimmedReply, options), 1);
            if isempty(idx)
                error('Invalid %s selection: %s', varName, reply);
            end
            value = options{idx};
        end
    else
        value = options{defaultIdx};
    end

    assignin('base', varName, value);
end
