function [M0, M1] = continuousMomentFromPolyline(times, amps)
    % Continuous M0 and M1 about t=0 for a piecewise-linear waveform.
    times = times(:).';
    amps = amps(:).';
    if numel(times) ~= numel(amps)
        error('times and amps must have the same length.');
    end
    M0 = 0;
    M1 = 0;
    for ii = 1:(numel(times)-1)
        t0 = times(ii);
        t1 = times(ii+1);
        g0 = amps(ii);
        g1 = amps(ii+1);
        if t1 <= t0
            error('times must be strictly increasing.');
        end
        m = (g1 - g0) / (t1 - t0);
        b = g0 - m*t0;
        M0 = M0 + 0.5*m*(t1^2 - t0^2) + b*(t1 - t0);
        M1 = M1 + (m/3)*(t1^3 - t0^3) + 0.5*b*(t1^2 - t0^2);
    end
end
