function [gpe_wave_sin, gpe_post] = defineSineWaveGradient4Calib(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpePre, gro, adc, physical_slew_max, waveInfoFlag, debugFlag)
    %DEFINECOSINEWAVEGRADIENT Create PE sine wave gradient with pre/post compensation.
    %
    % Outputs
    %   gpe_wave_sin : PE prephaser + rampup + sine wave gradient
    %   gpe_post     : PE rampdown/post-rewinder gradient
    %
    % Inputs
    %   Tread      : sine wave duration, usually ADC/readout duration [s]
    %   sys        : Pulseq system struct
    %   Ncycles    : number of sine cycles during Tread
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

    % Design sine wave
    wavepoints_sin = round(Tread / sys.gradRasterTime);
    tWaveUnit_sin = sys.gradRasterTime;

    TreadRaster = wavepoints_sin * tWaveUnit_sin;

    tWavePeriod_sin = TreadRaster / Ncycles;
    w_sin = 2*pi / tWavePeriod_sin;
    if waveInfoFlag
        fprintf('w_sin: %.6f rad/s, tWavePeriod_sin: %.6f ms\n', ...
        w_sin, tWavePeriod_sin*1e3);
    end

    % Determine sine amplitude from gradient and slew limits
    % Pulseq internal gradient units are Hz/m.
    % Reference design uses G/cm, then converts to Hz/m.
    swave_max = min(physical_slew_max, swave_max);

    swave_max_gauss = swave_max * 100;   % T/m/s -> G/cm/s
    gwave_max_gauss = gwave_max / 10;    % mT/m -> G/cm

    if swave_max_gauss >= w_sin * gwave_max_gauss
        G0_sin = gwave_max_gauss;
        if waveInfoFlag
            disp(['wave amplitude is not slew limited, using g0_sin = ', ...
                num2str(G0_sin*10), ' mT/m']);
        end
    else
        G0_sin = swave_max_gauss / w_sin;
        if waveInfoFlag
            disp(['wave amplitude is slew limited, using g0_sin = ', ...
                num2str(G0_sin*10), ' mT/m']);
        end
    end

    % Convert amplitude from G/cm to Pulseq Hz/m
    scaling_factor = sys.gamma * 1e-2;
    G0_sin_pulseq  = G0_sin * scaling_factor;

    % Build sine waveform
    tWavepoints_sin = (0:wavepoints_sin) * tWaveUnit_sin;  %TODO: minus 1 or not?
    gWave_sin = G0_sin_pulseq * sin(w_sin * tWavepoints_sin);
    
    % cover the extra dead-time region too.
    targetCosDur = adc.numSamples * adc.dwell + sys.adcDeadTime;
    % Current waveform duration
    currentCosDur = numel(gWave_sin) * sys.gradRasterTime;
    % Add constant samples at G0_sin_pulseq if needed
    nPad = round((targetCosDur - currentCosDur) / sys.gradRasterTime);
    if nPad > 0
        gWave_sin = [gWave_sin, zeros(1, nPad)];
    elseif nPad < 0
        error('nPad smaller than 0')
    end
    % Pad the front
    nPadPre = round(gro.riseTime / sys.gradRasterTime);
    gWave_sin = [zeros(1, nPadPre) gWave_sin];
    gWave_sin = gWave_sin(:).';

    gWave_sin_helper = mr.makeArbitraryGrad(gpe_channel, gWave_sin, 'system', sys, 'first', 0, 'last',  0);

    % Design merged PE prephaser + wave
    % Extract the waveform from gpePre
    tCorners = [0, gpePre.riseTime, gpePre.riseTime + gpePre.flatTime, gpePre.riseTime + gpePre.flatTime + gpePre.fallTime] + gpePre.delay;
    aCorners = [0, gpePre.amplitude, gpePre.amplitude, 0];
    % Filter out duplicates (crucial if the blip is perfectly triangular)
    [tCorners_unq, idx_unq] = unique(tCorners, 'stable');
    aCorners_unq = aCorners(idx_unq);
    % Sample the blip onto the raster grid
    dt = sys.gradRasterTime;
    n = round(gpePre_dur / dt);
    tCenters = ((0:n-1) + 0.5) * dt;
    preWave = interp1(tCorners_unq, aCorners_unq, tCenters, 'linear', 0);
    % Force row vector
    preWave = preWave(:).';

    % Concatenate prephaser and sine manually.
    % This avoids mr.addGradients row/column zero-fill issues.
    gpeWaveFull = [preWave, gWave_sin];
    gpe_wave_sin = mr.makeArbitraryGrad(gpe_channel, gpeWaveFull, 'system', sys, 'first', 0, 'last', 0);

    % Nominal post-rewinder area is just -gpePre.area.
    % Then compensate the residual area of the sine waveform.
    gpePost_area_new = -gpe_wave_sin.area;

    gpe_post = mr.makeTrapezoid(gpe_channel, 'Area', gpePost_area_new, 'system', sys_lowPNS);

    % Sanity check of time
    tol = sys.gradRasterTime/10;
    adcEndObject = mr.calcDuration(adc);
    fullDur_obj    = mr.calcDuration(gpe_wave_sin);
    if abs(adcEndObject - fullDur_obj) > tol
        error(['Timing mismatch: ADC duration (including delay) = %.6f ms, ', ...
            'Wave object (including prephase) = %.6f ms, diff = %.6f us'], ...
            adcEndObject*1e3, ...
            fullDur_obj*1e3, ...
            (adcEndObject - fullDur_wobj)*1e6);
    end

    % Debug print based on actual constructed objects
    if debugFlag

        dt = sys.gradRasterTime;

        % Actual durations from generated waveform arrays
        preRampDur_wave = numel(preWave) * dt;
        sinDur_wave     = numel(gWave_sin) * dt;
        fullDur_wave    = numel(gpeWaveFull) * dt;

        % Actual durations from Pulseq objects
        sinDur_obj     = mr.calcDuration(gWave_sin_helper);
        fullDur_obj    = mr.calcDuration(gpe_wave_sin);
        postDur_obj    = mr.calcDuration(gpe_post);

        % ADC timing from actual ADC object
        adcStart     = adc.delay;
        adcAcqDur    = adc.numSamples * adc.dwell;
        adcEndAcq    = adcStart + adcAcqDur;
        adcEndObject = mr.calcDuration(adc);

        % Sine timing inside the final concatenated PE waveform
        sinStart_actual = preRampDur_wave;
        sinEnd_actual   = fullDur_wave;

        fprintf('\n');
        fprintf('================ Sine PE wave debug ================\n');

        fprintf('gpePre.area                          = %.9g 1/m\n', gpePre.area);
        fprintf('gpePre duration                      = %.6f ms\n', gpePre_dur*1e3);
        fprintf('gro.riseTime                         = %.6f ms\n', gro.riseTime*1e3);

        fprintf('\n--- Pre-ramp timing ---\n');
        fprintf('preRampWave samples                  = %d\n', numel(preWave));
        fprintf('preRamp duration from waveform       = %.6f ms\n', preRampDur_wave*1e3);
        fprintf('gpePre duration + gro.riseTime       = %.6f ms\n', ...
            (gpePre_dur + gro.riseTime)*1e3);
        fprintf('preRampWave duration - adc.delay     = %.6f us\n', ...
            (preRampDur_wave - adcStart)*1e6);

        fprintf('\n--- Sine wave timing ---\n');
        fprintf('gWave_sin samples                    = %d\n', numel(gWave_sin));
        fprintf('sin duration from waveform           = %.6f ms\n', sinDur_wave*1e3);
        fprintf('sin duration from helper object      = %.6f ms\n', sinDur_obj*1e3);
        fprintf('Tread input                          = %.6f ms\n', Tread*1e3);
        fprintf('Tread rasterized                     = %.6f ms\n', TreadRaster*1e3);

        fprintf('\n--- ADC timing ---\n');
        fprintf('adc.delay                            = %.6f ms\n', adcStart*1e3);
        fprintf('adc.numSamples * adc.dwell           = %.6f ms\n', adcAcqDur*1e3);
        fprintf('adc acquisition end                  = %.6f ms\n', adcEndAcq*1e3);
        fprintf('mr.calcDuration(adc)                 = %.6f ms\n', adcEndObject*1e3);

        fprintf('\n--- Final PE wave timing ---\n');
        fprintf('gpe_wave_sin duration from waveform  = %.6f ms\n', fullDur_wave*1e3);
        fprintf('gpe_wave_sin duration from object    = %.6f ms\n', fullDur_obj*1e3);
        fprintf('gpe_post duration                    = %.6f ms\n', postDur_obj*1e3);

        fprintf('\n--- Timing differences ---\n');
        fprintf('full PE wave - mr.calcDuration(obj)  = %.6f us\n', ...
            (fullDur_wave - fullDur_obj)*1e6);

        fprintf('\n--- Areas ---\n');
        fprintf('preRamp object area                  = %.9g 1/m\n', gpePre.area);
        fprintf('sin helper area                      = %.9g 1/m\n', gWave_sin_helper.area);
        fprintf('full gpe_wave_sin area               = %.9g 1/m\n', gpe_wave_sin.area);
        fprintf('gpe_post target area                 = %.9g 1/m\n', gpePost_area_new);
        fprintf('gpe_post actual area                 = %.9g 1/m\n', gpe_post.area);

        fprintf('======================================================\n\n');

    end
end
