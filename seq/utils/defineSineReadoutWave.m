function [gWave, area, readWaveM1] = defineSineReadoutWave(channel, Tread, T_pre, T_total, sys, Ncycles, gwave_max, swave_max, physical_slew_max, adc, echoIndex, waveInfoFlag, debugFlag)
    dt = sys.gradRasterTime;
    [G0, w, TreadRaster] = designWaveAmplitude(Tread, sys, Ncycles, gwave_max, swave_max, physical_slew_max, 'sine', waveInfoFlag);

    nPre = round(T_pre / dt);
    nReadIntervals = round(TreadRaster / dt);

    % Include the endpoint sample at t = TreadRaster so the sine completes
    % the integer number of cycles cleanly. The ADC samples occupy the first
    % TreadRaster seconds of this wave segment; the endpoint sample is part
    % of the post-ADC block envelope.
    tRead = (0:nReadIntervals) * dt;
    waveRead = G0 * sin(w * tRead);
    nReadWave = numel(waveRead);

    % First moment of the sine wave part only. Do not include the zero
    % pre-padding or post-padding used to align the sine readout object.
    % Since the integer-cycle sine readout has nominally zero M0, this M1 is
    % translation-invariant up to raster/endpoint numerical residuals.
    readWaveM1 = sum(tRead .* waveRead) * dt;

    nTotal = round(T_total / dt);
    nPost = nTotal - nPre - nReadWave;
    if nPost < 0
        error(['Sine wave timing is longer than the requested block envelope. ', ...
               'Increase T_total or reduce T_pre/Tread.']);
    end

    waveFull = [zeros(1, nPre), waveRead, zeros(1, nPost)];
    waveFull = forceLength(waveFull, nTotal);

    gWave = mr.makeArbitraryGrad(channel, waveFull, 'system', sys, 'first', 0, 'last', 0);
    area = gWave.area;

    if debugFlag
        adcAcqDur = adc.numSamples * adc.dwell;
        fprintf('\n================ Sine readout wave debug: echo %d ================\n', echoIndex);
        fprintf('channel                              = %s\n', channel);
        fprintf('samples pre/readWave/post/total      = %d / %d / %d / %d\n', nPre, nReadWave, nPost, nTotal);
        fprintf('T_pre input                          = %.6f ms\n', T_pre*1e3);
        fprintf('Tread input/raster                   = %.6f / %.6f ms\n', Tread*1e3, TreadRaster*1e3);
        fprintf('sine read-wave duration incl endpoint= %.6f ms\n', nReadWave*dt*1e3);
        fprintf('T_total envelope                     = %.6f ms\n', T_total*1e3);
        fprintf('adc.delay                            = %.6f ms\n', adc.delay*1e3);
        fprintf('adc acquisition duration             = %.6f ms\n', adcAcqDur*1e3);
        fprintf('preWave end - adc.delay              = %.6f us\n', (nPre*dt - adc.delay)*1e6);
        fprintf('max amplitude                        = %.6f kHz/m\n', max(abs(waveFull))*1e-3);
        fprintf('wave object duration                 = %.6f ms\n', mr.calcDuration(gWave)*1e3);
        fprintf('wave area from object                = %.9g 1/m\n', area);
        fprintf('sine readout-wave M1 only            = %.9g 1/m*s\n', readWaveM1);
        fprintf('==================================================================\n\n');
    end
end
