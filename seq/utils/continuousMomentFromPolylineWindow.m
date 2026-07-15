function [M0, M1] = continuousMomentFromPolylineWindow(times, amps, tStart, tEnd, tRef)
    % Continuous M0 and M1 for a piecewise-linear waveform clipped to
    % [tStart, tEnd], with M1 referenced to tRef:
    %   M1 = int_{tStart}^{tEnd} (t - tRef) G(t) dt.
    if tEnd <= tStart
        M0 = 0;
        M1 = 0;
        return;
    end
    times = times(:).';
    amps = amps(:).';
    [times, idxUnq] = unique(times, 'stable');
    amps = amps(idxUnq);
    if numel(times) < 2
        M0 = 0;
        M1 = 0;
        return;
    end
    tInside = times(times > tStart & times < tEnd);
    tClip = unique([tStart, tInside, tEnd]);
    aClip = interp1(times, amps, tClip, 'linear', 0);
    [M0abs, M1abs] = continuousMomentFromPolyline(tClip, aClip);
    M0 = M0abs;
    M1 = M1abs - tRef * M0abs;
end
