function [rampWave, nRamp, T_ramp, slewPeak] = makeShortestEndpointRampWave(G_start, G_end, sys, minDuration)
    %MAKE_SHORTEST_ENDPOINT_RAMP_WAVE Shortest endpoint ramp sampled on the gradient raster.
    % The returned waveform includes both endpoint samples. Therefore the
    % number of slew intervals is nRamp-1, and the slew check is based on
    % diff(rampWave)/dt. An optional minDuration extends the ramp without
    % changing its endpoints. Units are Pulseq internal Hz/m.
    dt = sys.gradRasterTime;
    dG = G_end - G_start;

    if nargin < 4 || isempty(minDuration)
        minDuration = dt;
    end
    if ~isscalar(minDuration) || ~isfinite(minDuration) || minDuration < 0
        error('minDuration must be a finite nonnegative scalar.');
    end
    nMinSamples = max(1, ceil((minDuration-dt/10)/dt));

    if max(abs([G_start, G_end])) > sys.maxGrad + 1e-9
        error('Endpoint ramp amplitude exceeds system maxGrad.');
    end

    if abs(dG) < eps
        nRamp = nMinSamples;
        rampWave = G_end*ones(1, nRamp);
        T_ramp = nRamp*dt;
        slewPeak = 0;
        return;
    end

    nIntervals = max(1, ceil(abs(dG) / (sys.maxSlew * dt)));
    nRamp = max(nIntervals + 1, nMinSamples);
    rampWave = linspace(G_start, G_end, nRamp);
    slewPeak = max(abs(diff(rampWave))) / dt;

    if slewPeak > sys.maxSlew * (1 + 1e-9)
        error('Internal error: endpoint post-ramp exceeds slew limit after rasterization.');
    end

    T_ramp = nRamp * dt;
    if T_ramp + dt/10 < minDuration
        error('Internal error: endpoint ramp is shorter than minDuration.');
    end
end
