function [gWave, area, readWaveM1, timing] = ...
    defineSineReadoutWaveWithMoments(channel, Tread, T_pre, T_total, ...
    sys_wave, Ncycles, gwave_max, swave_max, physical_slew_max, adc, ...
    echoIndex, waveInfoFlag, debugFlag)
% Sine active waveform plus its exact serialized timing representation.

dt = sys_wave.gradRasterTime;
[G0, w, TreadRaster] = designWaveAmplitude(Tread, sys_wave, Ncycles, ...
    gwave_max, swave_max, physical_slew_max, 'sine', waveInfoFlag);
nPre = round(T_pre/dt);
nReadIntervals = round(TreadRaster/dt);
tRead = (0:nReadIntervals)*dt;
waveRead = G0*sin(w*tRead);
nReadWave = numel(waveRead);
readWaveM1 = sum(tRead.*waveRead)*dt;

nTotal = round(T_total/dt);
nPost = nTotal-nPre-nReadWave;
if nPost < 0
    error('Sine wave exceeds the requested block envelope.');
end
waveFull = [zeros(1, nPre), waveRead, zeros(1, nPost)];
waveFull = forceLength(waveFull, nTotal);
gWave = mr.makeArbitraryGrad(channel, waveFull, ...
    'system', sys_wave, 'first', 0, 'last', 0);
area = gWave.area;

timing = struct;
timing.G0 = G0;
timing.nPre = nPre;
timing.nReadWave = nReadWave;
timing.nPost = nPost;
timing.nTotal = nTotal;
timing.waveform = waveFull;
timing.sampleTimes = ((1:nTotal)-0.5)*dt;
timing.first = 0;
timing.last = 0;
timing.shapeDur = nTotal*dt;

if debugFlag
    fprintf(['\nSine echo %d: cycles=%d, G0=%.6f kHz/m, ', ...
        'area=%+.9g 1/m, active M1=%+.9g 1/m*s, ', ...
        'ADC center=%.6f ms.\n'], ...
        echoIndex, Ncycles, G0*1e-3, area, readWaveM1, ...
        (adc.delay+0.5*adc.numSamples*adc.dwell)*1e3);
end
end
