function [gWaveFull, gPost] = defineCenteredSineWaveGradient4Calib( ...
    Tread, sys_wave, sys_lowPNS, Ncycles, gwave_max, swave_max, ...
    gpePre, sineOffsetArea, gro, adc, cosRampDownDuration, ...
    physical_slew_max)
% Calibration sine with parity-aware center offset merged into its PE prep.

dt = sys_wave.gradRasterTime;
[G0, w, TreadRaster] = designWaveAmplitude(Tread, sys_wave, Ncycles, ...
    gwave_max, swave_max, physical_slew_max, 'calibration sine', false);
nIntervals = round(TreadRaster/dt);
waveSin = G0*sin(w*(0:nIntervals)*dt);
targetDur = adc.numSamples*adc.dwell+sys_wave.adcDeadTime;
nPad = round((targetDur-numel(waveSin)*dt)/dt);
assert(nPad >= 0, 'Calibration sine padding is negative.');
waveSin = [waveSin, zeros(1, nPad)];

combinedArea = gpePre.area+sineOffsetArea;
% Match the accepted standalone construction: when the offset is nonzero,
% PE and offset are one low-PNS trapezoid spanning the complete ADC delay,
% including the native readout-rise interval. For an exactly zero offset,
% retain the baseline PE followed by the readout-rise zero padding.
hasOffset = abs(sineOffsetArea) > ...
    max(1e-12, eps(max(1, abs(sineOffsetArea))));
if hasOffset
    preDuration = adc.delay;
    activePrePad = [];
    pre = mr.makeTrapezoid(gpePre.channel, 'Area', combinedArea, ...
        'Duration', preDuration, 'system', sys_lowPNS);
else
    preDuration = mr.calcDuration(gpePre);
    activePrePad = zeros(1, round(gro.riseTime/dt));
    % Preserve the scaled maximum-PE template exactly. Reconstructing an
    % equal-area/equal-duration trapezoid can select different ramp times.
    pre = gpePre;
end
[tCorners, aCorners] = trapezoidCorners(pre);
nPre = round(mr.calcDuration(pre)/dt);
tCenters = ((0:nPre-1)+0.5)*dt;
preWave = interp1(tCorners, aCorners, tCenters, 'linear', 0);
composite = [preWave(:).', activePrePad, waveSin];
gWaveFull = mr.makeArbitraryGrad(gpePre.channel, composite, ...
    'system', sys_wave, 'first', 0, 'last', 0);
postArea = -gWaveFull.area;
shortestPost = mr.makeTrapezoid(gpePre.channel, 'Area', postArea, ...
    'system', sys_lowPNS);
if hasOffset && ...
        cosRampDownDuration+dt/10 >= mr.calcDuration(shortestPost)
    gPost = mr.makeTrapezoid(gpePre.channel, 'Area', postArea, ...
        'Duration', cosRampDownDuration, 'system', sys_lowPNS);
else
    gPost = shortestPost;
end
assert(abs(mr.calcDuration(gWaveFull)-mr.calcDuration(adc)) < dt/10, ...
    'Calibration sine and ADC durations differ.');
end
