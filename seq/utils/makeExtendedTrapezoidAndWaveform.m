function [gExt, wave] = makeExtendedTrapezoidAndWaveform(channel, times, amps, T_total, sys)
    dt = sys.gradRasterTime;
    gExt = mr.makeExtendedTrapezoid(channel, 'times', times, 'amplitudes', amps, 'system', sys);
    n = round(T_total / dt);
    tCenters = ((0:n-1) + 0.5) * dt;
    wave = interp1(times, amps, tCenters, 'linear', 'extrap');
    wave = wave(:).';
end
