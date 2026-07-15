function [gpe_wave_cos, gpe_post] = defineCosineWaveGradient4Calib(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpePre, gro, adc, physical_slew_max, waveInfoFlag, debugFlag)
    %DEFINECOSINEWAVEGRADIENT Create PE cosine wave gradient with pre/post compensation.
    %
    % Outputs
    %   gpe_wave_cos : PE prephaser + rampup + cosine wave gradient
    %   gpe_post     : PE rampdown/post-rewinder gradient
    %
    % Inputs
    %   Tread      : cosine wave duration, usually ADC/readout duration [s]
    %   sys        : Pulseq system struct
    %   Ncycles    : number of cosine cycles during Tread
    %   gwave_max  : max wave gradient amplitude [mT/m]
    %   swave_max  : max wave slew rate [T/m/s]
    %   gpePre     : nominal PE prephaser gradient
    %   gro        : readout gradient; gro.riseTime is used for rampup
    if nargin < 10
        debugFlag = false;
        waveInfoFlag = false;
    end

    % Basic checks
    if Ncycles <= 0
        error('Ncycles must be positive.');
    end

    if Tread <= 0
        error('Tread must be positive.');
    end

    % Get PE channel and prephaser duration from gpePre
    gpe_channel = gpePre.channel;
    gpePre_dur  = mr.calcDuration(gpePre);

    % Design cosine wave
    wavepoints_cos = round(Tread / sys.gradRasterTime);
    tWaveUnit_cos  = sys.gradRasterTime;

    TreadRaster = wavepoints_cos * tWaveUnit_cos;

    tWavePeriod_cos = TreadRaster / Ncycles;
    w_cos = 2*pi / tWavePeriod_cos;
    if waveInfoFlag
        fprintf('w_cos: %.6f rad/s, tWavePeriod_cos: %.6f ms\n', ...
        w_cos, tWavePeriod_cos*1e3);
    end

    % Determine cosine amplitude from gradient and slew limits
    % Pulseq internal gradient units are Hz/m.
    % Reference design uses G/cm, then converts to Hz/m.
    swave_max = min(physical_slew_max, swave_max);

    swave_max_gauss = swave_max * 100;   % T/m/s -> G/cm/s
    gwave_max_gauss = gwave_max / 10;    % mT/m -> G/cm

    if swave_max_gauss >= w_cos * gwave_max_gauss
        G0_cos = gwave_max_gauss;
        if waveInfoFlag
            disp(['wave amplitude is not slew limited, using g0_cos = ', ...
                num2str(G0_cos*10), ' mT/m']);
        end
    else
        G0_cos = swave_max_gauss / w_cos;
        if waveInfoFlag
            disp(['wave amplitude is slew limited, using g0_cos = ', ...
                num2str(G0_cos*10), ' mT/m']);
        end
    end

    % Convert amplitude from G/cm to Pulseq Hz/m
    scaling_factor = sys.gamma * 1e-2;
    G0_cos_pulseq  = G0_cos * scaling_factor;

    % Build cosine waveform
    tWavepoints_cos = ((0:wavepoints_cos)) * tWaveUnit_cos;  %TODO: plus 0.5 or not?
    gWave_cos = G0_cos_pulseq * cos(w_cos * tWavepoints_cos);
    
    % cover the extra dead-time region too.
    targetCosDur = adc.numSamples * adc.dwell + sys.adcDeadTime;
    % Current waveform duration
    currentCosDur = numel(gWave_cos) * sys.gradRasterTime;
    % Add constant samples at G0_cos_pulseq if needed
    nPad = round((targetCosDur - currentCosDur) / sys.gradRasterTime);
    if nPad > 0
        gWave_cos = [gWave_cos, G0_cos_pulseq * ones(1, nPad)];
    elseif nPad < 0
        error('nPad smaller than 0')
    end
    gWave_cos = gWave_cos(:).';

    gWave_cos_helper = mr.makeArbitraryGrad(gpe_channel, gWave_cos, 'system', sys, 'first', G0_cos_pulseq, 'last',  G0_cos_pulseq);
    % Design rampdown/post compensation
    gWave_cos_area_max_unit = gWave_cos_helper.area / (nPad + 1);
    if waveInfoFlag
        disp(['gWave_cos_area_max_unit: ' num2str(gWave_cos_area_max_unit)])
    end

    % Design merged PE prephaser + wave rampup
    % The merged event starts at 0, ends at G0_cos_pulseq, has total area gpePre.area, and lasts gpePre_dur + gro.riseTime.
    T_preRamp = gpePre_dur + gro.riseTime;
    T_preRamp = round(T_preRamp / sys.gradRasterTime) * sys.gradRasterTime;
    A_preRamp_target = gpePre.area - gWave_cos_area_max_unit / 2 / (sys.gradRasterTime / adc.dwell);
    G_rampup_end = G0_cos_pulseq;
    [gpePre_rampup, preRampWave] = makeFixedDurationPreRamp4Calib(gpe_channel, A_preRamp_target, G_rampup_end, T_preRamp, sys);

    % Concatenate pre-ramp and cosine manually.
    % This avoids mr.addGradients row/column zero-fill issues.
    gpeWaveFull = [preRampWave, gWave_cos];

    % gpe_wave_cos = mr.makeArbitraryGrad(gpe_channel, gpeWaveFull, 'system', sys, 'first', 0, 'last', G0_cos_pulseq, 'oversampling', true);  % has timing errors
    gpe_wave_cos = mr.makeArbitraryGrad(gpe_channel, gpeWaveFull, 'system', sys, 'first', 0, 'last', G0_cos_pulseq);

    % Nominal post-rewinder area is just -gpePre.area.
    % Then compensate the residual area of the cosine waveform.
    gpePost_area_new = -gpe_wave_cos.area;

    gpe_post = mr.makeExtendedTrapezoidArea(gpe_channel, G0_cos_pulseq, 0, gpePost_area_new, sys_lowPNS);

    % Sanity check of time
    tol = sys.gradRasterTime/10;
    adcEndObject = mr.calcDuration(adc);
    fullDur_obj    = mr.calcDuration(gpe_wave_cos);
    if abs(adcEndObject - fullDur_obj) > tol
        error(['Timing mismatch: ADC duration (including delay) = %.6f ms, ', ...
            'Wave object (including prephase) = %.6f ms, diff = %.6f us'], ...
            adcEndObject*1e3, ...
            fullDur_obj*1e3, ...
            (adcEndObject - fullDur_obj)*1e6);
    end

    % Debug print based on actual constructed objects
    if debugFlag

        dt = sys.gradRasterTime;

        % Actual durations from generated waveform arrays
        preRampDur_wave = numel(preRampWave) * dt;
        cosDur_wave     = numel(gWave_cos) * dt;
        fullDur_wave    = numel(gpeWaveFull) * dt;

        % Actual durations from Pulseq objects
        preRampDur_obj = mr.calcDuration(gpePre_rampup);
        cosDur_obj     = mr.calcDuration(gWave_cos_helper);
        fullDur_obj    = mr.calcDuration(gpe_wave_cos);
        postDur_obj    = mr.calcDuration(gpe_post);

        % ADC timing from actual ADC object
        adcStart     = adc.delay;
        adcAcqDur    = adc.numSamples * adc.dwell;
        adcEndAcq    = adcStart + adcAcqDur;
        adcEndObject = mr.calcDuration(adc);

        % Cosine timing inside the final concatenated PE waveform
        cosStart_actual = preRampDur_wave;
        cosEnd_actual   = fullDur_wave;

        fprintf('\n');
        fprintf('================ Cosine PE wave debug ================\n');

        fprintf('gpePre.area                         = %.9g 1/m\n', gpePre.area);
        fprintf('gpePre duration                     = %.6f ms\n', gpePre_dur*1e3);
        fprintf('gro.riseTime                        = %.6f ms\n', gro.riseTime*1e3);

        fprintf('\n--- Pre-ramp timing ---\n');
        fprintf('preRampWave samples                 = %d\n', numel(preRampWave));
        fprintf('preRamp duration from waveform       = %.6f ms\n', preRampDur_wave*1e3);
        fprintf('preRamp duration from object         = %.6f ms\n', preRampDur_obj*1e3);
        fprintf('gpePre duration + gro.riseTime       = %.6f ms\n', ...
            (gpePre_dur + gro.riseTime)*1e3);
        fprintf('preRampWave duration - adc.delay     = %.6f us\n', ...
            (preRampDur_wave - adcStart)*1e6);

        fprintf('\n--- Cosine wave timing ---\n');
        fprintf('gWave_cos samples                   = %d\n', numel(gWave_cos));
        fprintf('cos duration from waveform           = %.6f ms\n', cosDur_wave*1e3);
        fprintf('cos duration from helper object      = %.6f ms\n', cosDur_obj*1e3);
        fprintf('Tread input                          = %.6f ms\n', Tread*1e3);
        fprintf('Tread rasterized                     = %.6f ms\n', TreadRaster*1e3);

        fprintf('\n--- ADC timing ---\n');
        fprintf('adc.delay                            = %.6f ms\n', adcStart*1e3);
        fprintf('adc.numSamples * adc.dwell           = %.6f ms\n', adcAcqDur*1e3);
        fprintf('adc acquisition end                  = %.6f ms\n', adcEndAcq*1e3);
        fprintf('mr.calcDuration(adc)                 = %.6f ms\n', adcEndObject*1e3);

        fprintf('\n--- Final PE wave timing ---\n');
        fprintf('cosine starts at                     = %.6f ms\n', cosStart_actual*1e3);
        fprintf('cosine ends at                       = %.6f ms\n', cosEnd_actual*1e3);
        fprintf('gpe_wave_cos duration from waveform  = %.6f ms\n', fullDur_wave*1e3);
        fprintf('gpe_wave_cos duration from object    = %.6f ms\n', fullDur_obj*1e3);
        fprintf('gpe_post duration                    = %.6f ms\n', postDur_obj*1e3);

        fprintf('\n--- Timing differences ---\n');
        fprintf('cos start - adc start                = %.6f us\n', ...
            (cosStart_actual - adcStart)*1e6);
        fprintf('cos end - adc acquisition end        = %.6f us\n', ...
            (cosEnd_actual - adcEndAcq)*1e6);
        fprintf('cos duration - adc acquisition dur   = %.6f us\n', ...
            (cosDur_wave - adcAcqDur)*1e6);
        fprintf('full PE wave - mr.calcDuration(obj)  = %.6f us\n', ...
            (fullDur_wave - fullDur_obj)*1e6);

        fprintf('\n--- Areas ---\n');
        fprintf('preRamp object area                  = %.9g 1/m\n', gpePre_rampup.area);
        fprintf('cos helper area                      = %.9g 1/m\n', gWave_cos_helper.area);
        fprintf('full gpe_wave_cos area               = %.9g 1/m\n', gpe_wave_cos.area);
        fprintf('gpe_post target area                 = %.9g 1/m\n', gpePost_area_new);
        fprintf('gpe_post actual area                 = %.9g 1/m\n', gpe_post.area);

        fprintf('======================================================\n\n');

    end
end
