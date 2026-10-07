function [gPreTrap, gWave, totalArea, newCarry, timing, rampUpM1] = ...
    defineParityCosineReadoutWave(channel, Tread, T_preTrap, ...
    T_wavePrePad, T_minTotal, sys_wave, sys_lowPNS, Ncycles, ...
    gwave_max, swave_max, physical_slew_max, adc, echoIndex, ...
    prevCarry, waveInfoFlag, debugFlag)
% Cosine readout body with low-PNS ramps and parity-aware ADC correction.

dt = sys_wave.gradRasterTime;
[G0, w, TreadRaster] = designWaveAmplitude(Tread, sys_wave, Ncycles, ...
    gwave_max, swave_max, physical_slew_max, 'cosine', ...
    waveInfoFlag && echoIndex == 1);
nReadIntervals = round(TreadRaster/dt);

[rampUpWave, nRampUp, T_rampUp, rampUpSlew] = ...
    makeShortestEndpointRampWave(0, G0, sys_lowPNS, T_wavePrePad);
rampUpTmp = mr.makeArbitraryGrad(channel, rampUpWave, ...
    'system', sys_wave, 'first', 0, 'last', G0);
A_rampUp = rampUpTmp.area;
tRampUp = (0:(numel(rampUpWave)-1))*dt;
rampUpM1 = sum(tRampUp.*rampUpWave)*dt;
assert(abs(T_rampUp-T_wavePrePad) < dt/10, ...
    'Cosine ramp-up and module pre-padding differ.');

tRead = (0:nReadIntervals)*dt;
waveRead = G0*cos(w*tRead);
nReadWave = numel(waveRead);
[postRampWave, nPostRamp, T_postRamp, postRampSlew] = ...
    makeShortestEndpointRampWave(waveRead(end), 0, sys_lowPNS);

nMinTotal = round(T_minTotal/dt);
nBaseTotal = nRampUp+nReadWave+nPostRamp;
nPostZeroPad = max(0, nMinTotal-nBaseTotal);
waveFull = [rampUpWave, waveRead, postRampWave, ...
    zeros(1, nPostZeroPad)];
nTotal = numel(waveFull);
gWave = mr.makeArbitraryGrad(channel, waveFull, ...
    'system', sys_wave, 'first', 0, 'last', 0);

centerPolarity = (-1)^Ncycles;
A_adcCenterCorr = -centerPolarity*0.5*G0*adc.dwell;
A_pre_target_total = A_adcCenterCorr-prevCarry;
A_pre_trap = A_pre_target_total-A_rampUp;
gPreTrapNatural = mr.makeTrapezoid(channel, 'Area', A_pre_trap, ...
    'system', sys_lowPNS);
T_preTrapNatural = ceil(mr.calcDuration(gPreTrapNatural)/dt)*dt;
if isnan(T_preTrap)
    gPreTrap = gPreTrapNatural;
else
    T_preTrap = round(T_preTrap/dt)*dt;
    if T_preTrap+dt/10 < T_preTrapNatural
        error(['Requested cosine prep duration %.6f ms is shorter than ', ...
            'natural minimum %.6f ms for echo %d.'], ...
            T_preTrap*1e3, T_preTrapNatural*1e3, echoIndex);
    end
    gPreTrap = mr.makeTrapezoid(channel, 'Area', A_pre_trap, ...
        'Duration', T_preTrap, 'system', sys_lowPNS);
end
T_preTrapActual = mr.calcDuration(gPreTrap);

totalArea = gPreTrap.area+gWave.area;
newCarry = prevCarry+totalArea;

timing = struct;
timing.nRampUp = nRampUp;
timing.nReadWave = nReadWave;
timing.nPostRamp = nPostRamp;
timing.nPostZeroPad = nPostZeroPad;
timing.nTotal = nTotal;
timing.TrampUp = T_rampUp;
timing.TreadWave = nReadWave*dt;
timing.TpostRamp = T_postRamp;
timing.Ttotal = nTotal*dt;
timing.rampUpSlew = rampUpSlew;
timing.postRampSlew = postRampSlew;
timing.G0 = G0;
timing.centerPolarity = centerPolarity;
timing.adcCenterCorrection = A_adcCenterCorr;
timing.rampUpArea = A_rampUp;
timing.rampUpM1 = rampUpM1;
timing.readoutArea = gWave.area;
timing.preTrapAreaTarget = A_pre_trap;
timing.preTrapAreaActual = gPreTrap.area;
timing.preTrapNaturalDur = T_preTrapNatural;
timing.preTrapActualDur = T_preTrapActual;
timing.waveform = waveFull;
timing.sampleTimes = ((1:nTotal)-0.5)*dt;
timing.first = 0;
timing.last = 0;
timing.shapeDur = nTotal*dt;

if debugFlag
    fprintf(['\nParity cosine echo %d: cycles=%d, polarity=%+d, ', ...
        'G0=%.6f kHz/m, ADC corr=%+.9g 1/m, ', ...
        'pre=%+.9g, wave=%+.9g, carry=%+.9g 1/m; ', ...
        'ramp slew up/down=%.6f/%.6f T/m/s.\n'], ...
        echoIndex, Ncycles, centerPolarity, G0*1e-3, ...
        A_adcCenterCorr, gPreTrap.area, gWave.area, newCarry, ...
        rampUpSlew/sys_wave.gamma, postRampSlew/sys_wave.gamma);
end
end
