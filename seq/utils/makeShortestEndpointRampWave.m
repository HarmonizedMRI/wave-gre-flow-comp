function [rampWave, nRamp, T_ramp, slewPeak] = makeShortestEndpointRampWave(G_start, G_end, sys)
    %MAKE_SHORTEST_ENDPOINT_RAMP_WAVE Shortest endpoint ramp sampled on the gradient raster.
    % The returned waveform includes both endpoint samples. Therefore the
    % number of slew intervals is nRamp-1, and the slew check is based on
    % diff(rampWave)/dt. Units are Pulseq internal Hz/m.
    dt = sys.gradRasterTime;
    dG = G_end - G_start;

    if max(abs([G_start, G_end])) > sys.maxGrad + 1e-9
        error('Endpoint ramp amplitude exceeds system maxGrad.');
    end

    if abs(dG) < eps
        rampWave = G_end;
        nRamp = 1;
        T_ramp = dt;
        slewPeak = 0;
        return;
    end

    nIntervals = max(1, ceil(abs(dG) / (sys.maxSlew * dt)));
    nRamp = nIntervals + 1;
    rampWave = linspace(G_start, G_end, nRamp);
    slewPeak = max(abs(diff(rampWave))) / dt;

    if slewPeak > sys.maxSlew * (1 + 1e-9)
        error('Internal error: endpoint post-ramp exceeds slew limit after rasterization.');
    end

    T_ramp = nRamp * dt;
end
