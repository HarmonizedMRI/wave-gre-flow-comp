function [gExt, wave] = makeExtendedTrapezoidAndWaveform4Calib( ...
    channel, times, amps, T_total, sys)

    dt = sys.gradRasterTime;

    % Make the actual extended trapezoid object
    gExt = mr.makeExtendedTrapezoid(channel, ...
        'times', times, ...
        'amplitudes', amps, ...
        'system', sys);

    % Convert the same edge-defined waveform to Pulseq center-sampled waveform
    n = round(T_total / dt);

    tCenters = ((0:n-1) + 0.5) * dt;

    wave = interp1(times, amps, tCenters, 'linear');

    % Force row vector
    wave = wave(:).';

end
