function tf = promptYesNoFromWorkspace(varName, promptText, defaultValue)
    if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        workspaceValue = evalin('base', varName);
        if islogical(workspaceValue) || isnumeric(workspaceValue)
            tf = logical(workspaceValue);
            return;
        elseif ischar(workspaceValue) || isstring(workspaceValue)
            txt = lower(strtrim(char(workspaceValue)));
            if any(strcmp(txt, {'y', 'yes', 'true', '1'}))
                tf = true;
                return;
            elseif any(strcmp(txt, {'n', 'no', 'false', '0'}))
                tf = false;
                return;
            end
        end
        warning('Ignoring invalid workspace value for %s.', varName);
    end

    if defaultValue
        suffix = '[Y/n]';
    else
        suffix = '[y/N]';
    end

    reply = input(sprintf('%s %s: ', promptText, suffix), 's');
    reply = lower(strtrim(reply));
    if isempty(reply)
        tf = logical(defaultValue);
    elseif any(strcmp(reply, {'y', 'yes', 'true', '1'}))
        tf = true;
    elseif any(strcmp(reply, {'n', 'no', 'false', '0'}))
        tf = false;
    else
        error('Please answer yes or no for %s.', promptText);
    end
    assignin('base', varName, tf);
end
