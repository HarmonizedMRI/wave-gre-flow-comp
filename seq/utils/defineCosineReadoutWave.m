function [gPreTrap, gWave, totalArea, newCarry, timing, rampUpM1] = defineCosineReadoutWave(channel, Tread, T_preTrap, T_wavePrePad, T_minTotal, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, physical_slew_max, adc, echoIndex, prevCarry, waveInfoFlag, debugFlag)
    dt = sys.gradRasterTime;
    [G0, w, TreadRaster] = designWaveAmplitude(Tread, sys, Ncycles, gwave_max, swave_max, physical_slew_max, 'cosine', waveInfoFlag && echoIndex == 1);

    nReadIntervals = round(TreadRaster / dt);

    % The cosine readout body contains only the minimum-time 0 -> G0 ramp,
    % the ADC-window cosine, and the explicit G0 -> 0 ramp-down. The M0/carry
    % correction is a separate zero-endpoint trapezoid in the prep module.
    [rampUpWave, nRampUp, T_rampUp, rampUpSlew] = makeShortestEndpointRampWave(0, G0, sys_lowPNS);
    rampUpTmp = mr.makeArbitraryGrad(channel, rampUpWave, 'system', sys, 'first', 0, 'last', G0);
    A_rampUp = rampUpTmp.area;

    % Step 5 part 1 diagnostic: M1 of the cosine ramp-up only, using the
    % start of the readout/wave module as the reference time. This intentionally
    % uses the sampled rampUpWave vector, matching the staged sine M1 diagnostic.
    tRampUp = (0:(numel(rampUpWave)-1)) * dt;
    rampUpM1 = sum(tRampUp .* rampUpWave) * dt;

    assert(abs(T_rampUp - T_wavePrePad) < dt/10, ...
        'Cosine ramp-up duration %.6f ms does not match module pre-padding %.6f ms.', ...
        T_rampUp*1e3, T_wavePrePad*1e3);

    % Include the endpoint sample at t = TreadRaster. For an integer number
    % of cycles, the final cosine sample is exactly +G0, so the following
    % post-ramp is an explicit slew-safe G0 -> 0 segment.
    tRead = (0:nReadIntervals) * dt;
    waveRead = G0 * cos(w * tRead);
    nReadWave = numel(waveRead);

    [postRampWave, nPostRamp, T_postRamp, postRampSlew] = makeShortestEndpointRampWave(waveRead(end), 0, sys_lowPNS);

    nMinTotal = round(T_minTotal / dt);
    nBaseTotal = nRampUp + nReadWave + nPostRamp;
    nPostZeroPad = max(0, nMinTotal - nBaseTotal);

    waveFull = [rampUpWave, waveRead, postRampWave, zeros(1, nPostZeroPad)];
    nTotal = numel(waveFull);
    gWave = mr.makeArbitraryGrad(channel, waveFull, 'system', sys, 'first', 0, 'last', 0);

    % Base pre-trap area: Siemens ADC-center correction plus the negative of
    % the already accumulated realized cosine carry. The separate prep
    % trapezoid also cancels the positive ramp-up area because the ramp-up is
    % now inside the readout/wave module rather than hidden in the old
    % targeted endpoint pre-ramp.
    A_adcCenterCorr = -0.5 * G0 * adc.dwell;
    A_pre_target_total = A_adcCenterCorr - prevCarry;
    A_pre_trap = A_pre_target_total - A_rampUp;
    gPreTrapNatural = mr.makeTrapezoid(channel, 'Area', A_pre_trap, ...
        'system', sys_lowPNS);
    T_preTrapNatural = ceil(mr.calcDuration(gPreTrapNatural) / dt) * dt;
    if isnan(T_preTrap)
        gPreTrap = gPreTrapNatural;
    else
        T_preTrap = round(T_preTrap / dt) * dt;
        if T_preTrap + dt/10 < T_preTrapNatural
            error(['Requested cosine prep duration %.6f ms is shorter than ', ...
                   'natural minimum %.6f ms for echo %d.'], ...
                T_preTrap*1e3, T_preTrapNatural*1e3, echoIndex);
        end
        gPreTrap = mr.makeTrapezoid(channel, 'Area', A_pre_trap, ...
            'Duration', T_preTrap, 'system', sys_lowPNS);
    end
    T_preTrapActual = mr.calcDuration(gPreTrap);

    % Carry is taken from realized Pulseq objects that are actually played:
    % separate prep trapezoid plus the split cosine readout/wave body.
    totalArea = gPreTrap.area + gWave.area;
    newCarry = prevCarry + totalArea;

    timing = struct;
    timing.nRampUp = nRampUp;
    timing.nReadWave = nReadWave;
    timing.nPostRamp = nPostRamp;
    timing.nPostZeroPad = nPostZeroPad;
    timing.nTotal = nTotal;
    timing.TrampUp = T_rampUp;
    timing.TreadWave = nReadWave * dt;
    timing.TpostRamp = T_postRamp;
    timing.Ttotal = nTotal * dt;
    timing.rampUpSlew = rampUpSlew;
    timing.postRampSlew = postRampSlew;
    timing.G0 = G0;
    timing.rampUpArea = A_rampUp;
    timing.rampUpM1 = rampUpM1;
    timing.readoutArea = gWave.area;
    timing.preTrapAreaTarget = A_pre_trap;
    timing.preTrapAreaActual = gPreTrap.area;
    timing.preTrapNaturalDur = T_preTrapNatural;
    timing.preTrapActualDur = T_preTrapActual;

    % Preserve the exact arbitrary-gradient samples used to create gWave.
    % Pulseq arbitrary-gradient samples lie at raster-bin centers, with
    % explicit boundary values at t=0 and t=Ttotal.
    timing.waveform = waveFull;
    timing.sampleTimes = ((1:nTotal) - 0.5) * dt;
    timing.first = 0;
    timing.last = 0;
    timing.shapeDur = nTotal * dt;

    if debugFlag
        adcAcqDur = adc.numSamples * adc.dwell;
        fprintf('\n================ Split cosine readout wave debug: echo %d ================\n', echoIndex);
        fprintf('channel                                  = %s\n', channel);
        fprintf('samples rampUp/readWave/postRamp/zero/total = %d / %d / %d / %d / %d\n', ...
            nRampUp, nReadWave, nPostRamp, nPostZeroPad, nTotal);
        fprintf('prep trapezoid natural/actual duration   = %.6f / %.6f ms\n', T_preTrapNatural*1e3, T_preTrapActual*1e3);
        fprintf('ramp-up duration                         = %.6f ms\n', T_rampUp*1e3);
        fprintf('Tread input/raster                       = %.6f / %.6f ms\n', Tread*1e3, TreadRaster*1e3);
        fprintf('cos read-wave duration incl endpoint     = %.6f ms\n', nReadWave*dt*1e3);
        fprintf('minimum x-readout module envelope        = %.6f ms\n', T_minTotal*1e3);
        fprintf('actual cosine readout object duration    = %.6f ms\n', mr.calcDuration(gWave)*1e3);
        fprintf('adc.delay                                = %.6f ms\n', adc.delay*1e3);
        fprintf('adc acquisition duration                 = %.6f ms\n', adcAcqDur*1e3);
        fprintf('ramp-up end - adc.delay                  = %.6f us\n', (T_rampUp - adc.delay)*1e6);
        fprintf('G0                                       = %.6f kHz/m\n', G0*1e-3);
        fprintf('ramp-up peak slew                        = %.6f T/m/s equiv\n', rampUpSlew / sys.gamma);
        fprintf('post-ramp duration                       = %.6f us\n', T_postRamp*1e6);
        fprintf('post-ramp peak slew                      = %.6f T/m/s equiv\n', postRampSlew / sys.gamma);
        fprintf('A_adcCenterCorr                          = %.9g 1/m\n', A_adcCenterCorr);
        fprintf('prevCarry applied                        = %.9g 1/m\n', prevCarry);
        fprintf('ramp-up area                             = %.9g 1/m\n', A_rampUp);
        fprintf('ramp-up M1, ref=readout module start     = %.9g 1/m*s\n', rampUpM1);
        fprintf('prep target total area before ramp split = %.9g 1/m\n', A_pre_target_total);
        fprintf('prep trapezoid target area               = %.9g 1/m\n', A_pre_trap);
        fprintf('prep trapezoid actual area               = %.9g 1/m\n', gPreTrap.area);
        fprintf('cos readout object area                  = %.9g 1/m\n', gWave.area);
        fprintf('total cosine area this echo              = %.9g 1/m\n', totalArea);
        fprintf('new carry after this echo                = %.9g 1/m\n', newCarry);
        fprintf('========================================================================\n\n');
    end
end
