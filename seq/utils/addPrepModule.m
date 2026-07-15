function addPrepModule(seq, gxPrep, gzCosPre, gySineFC, echoIndex, isUseWaveCos, isUseSineFC, gyInitialFCByY, yIndex, isUseInitialYFC, gzInitialFCByZ, zIndex, isUseInitialZFC, gyInterEchoFCByY, isUseInterEchoPEYFC, gzInterEchoFCByZ, isUseInterEchoZFC)
    if nargin < 8
        gyInitialFCByY = {};
        yIndex = [];
        isUseInitialYFC = false;
    end
    if nargin < 11
        gzInitialFCByZ = {};
        zIndex = [];
        isUseInitialZFC = false;
    end
    if nargin < 14
        gyInterEchoFCByY = {};
        isUseInterEchoPEYFC = false;
    end
    if nargin < 16
        gzInterEchoFCByZ = {};
    end
    if nargin < 17
        isUseInterEchoZFC = false;
    end

    events = {gxPrep};
    if echoIndex == 1 && isUseInitialZFC && ~isempty(gzInitialFCByZ)
        if isempty(zIndex) || zIndex < 1 || zIndex > numel(gzInitialFCByZ)
            error('Invalid zIndex for initial PAR-dependent z FC module.');
        end
        events{end+1} = gzInitialFCByZ{zIndex}; %#ok<AGROW>
    elseif echoIndex > 1 && isUseInterEchoZFC && ~isempty(gzInterEchoFCByZ)
        if isempty(zIndex) || zIndex < 1 || zIndex > size(gzInterEchoFCByZ, 2)
            error('Invalid zIndex for inter-echo PAR-dependent z FC module.');
        end
        if isempty(gzInterEchoFCByZ{echoIndex, zIndex})
            error('Missing inter-echo PAR-dependent z FC module for echo %d, z index %d.', echoIndex, zIndex);
        end
        events{end+1} = gzInterEchoFCByZ{echoIndex, zIndex}; %#ok<AGROW>
    elseif isUseWaveCos
        events{end+1} = gzCosPre{echoIndex}; %#ok<AGROW>
    end
    if echoIndex == 1 && isUseInitialYFC && ~isempty(gyInitialFCByY)
        if isempty(yIndex) || yIndex < 1 || yIndex > numel(gyInitialFCByY)
            error('Invalid yIndex for initial PE-dependent y FC module.');
        end
        events{end+1} = gyInitialFCByY{yIndex}; %#ok<AGROW>
    elseif echoIndex > 1 && isUseInterEchoPEYFC && ~isempty(gyInterEchoFCByY)
        if isempty(yIndex) || yIndex < 1 || yIndex > size(gyInterEchoFCByY, 2)
            error('Invalid yIndex for inter-echo PE-dependent y FC module.');
        end
        if isempty(gyInterEchoFCByY{echoIndex, yIndex})
            error('Missing inter-echo PE-dependent y FC module for echo %d, y index %d.', echoIndex, yIndex);
        end
        events{end+1} = gyInterEchoFCByY{echoIndex, yIndex}; %#ok<AGROW>
    elseif isUseSineFC && numel(gySineFC) >= echoIndex && ~isempty(gySineFC{echoIndex})
        events{end+1} = gySineFC{echoIndex}; %#ok<AGROW>
    end
    seq.addBlock(events{:});
end
