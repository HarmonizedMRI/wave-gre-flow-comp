function [gWaveFull, gPost] = defineParityCosineWaveGradient4Calib( ...
    Tread, sys_wave, sys_lowPNS, Ncycles, gwave_max, swave_max, ...
    gpePre, gro, adc, physical_slew_max)
% Calibration cosine with low-PNS pre/post ramps and parity correction.

dt = sys_wave.gradRasterTime;
[G0, w, TreadRaster] = designWaveAmplitude(Tread, sys_wave, Ncycles, ...
    gwave_max, swave_max, physical_slew_max, 'calibration cosine', false);
nIntervals = round(TreadRaster/dt);
waveCos = G0*cos(w*(0:nIntervals)*dt);
targetDur = adc.numSamples*adc.dwell+sys_wave.adcDeadTime;
nPad = round((targetDur-numel(waveCos)*dt)/dt);
assert(nPad >= 0, 'Calibration cosine padding is negative.');
waveCos = [waveCos, G0*ones(1, nPad)];
cosObject = mr.makeArbitraryGrad(gpePre.channel, waveCos, ...
    'system', sys_wave, 'first', G0, 'last', G0);
areaUnit = cosObject.area/(nPad+1);

preDuration = mr.calcDuration(gpePre)+gro.riseTime;
centerPolarity = (-1)^Ncycles;
preArea = gpePre.area-centerPolarity*areaUnit/2 ...
    /(dt/adc.dwell);
[~, preWave] = makeFixedDurationPreRamp4Calib( ...
    gpePre.channel, preArea, G0, preDuration, sys_lowPNS);
composite = [preWave, waveCos];
gWaveFull = mr.makeArbitraryGrad(gpePre.channel, composite, ...
    'system', sys_wave, 'first', 0, 'last', G0);
gPost = mr.makeExtendedTrapezoidArea( ...
    gpePre.channel, G0, 0, -gWaveFull.area, sys_lowPNS);
assert(abs(mr.calcDuration(gWaveFull)-mr.calcDuration(adc)) < dt/10, ...
    'Calibration cosine and ADC durations differ.');
end
