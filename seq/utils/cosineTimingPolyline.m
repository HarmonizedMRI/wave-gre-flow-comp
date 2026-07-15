function [t, a] = cosineTimingPolyline(timing)
    %COSINETIMINGPOLYLINE Convert stored Pulseq arbitrary-gradient samples
    % to the continuous piecewise-linear representation used by the moment
    % integration helpers.
    required = {'waveform', 'sampleTimes', 'first', 'last', 'shapeDur'};
    for ii = 1:numel(required)
        if ~isfield(timing, required{ii})
            error('Cosine timing structure is missing field %s.', required{ii});
        end
    end

    waveform = timing.waveform(:).';
    sampleTimes = timing.sampleTimes(:).';
    if numel(waveform) ~= numel(sampleTimes)
        error('Cosine timing waveform/sampleTimes length mismatch.');
    end

    t = [0, sampleTimes, timing.shapeDur];
    a = [timing.first, waveform, timing.last];
    [t, idxUnique] = unique(t, 'stable');
    a = a(idxUnique);
end
