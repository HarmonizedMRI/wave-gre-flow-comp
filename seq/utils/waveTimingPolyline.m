function [t, a] = waveTimingPolyline(timing)
% Convert Pulseq center-sampled arbitrary-gradient timing to a polyline.

waveform = timing.waveform(:).';
sampleTimes = timing.sampleTimes(:).';
assert(numel(waveform) == numel(sampleTimes), ...
    'Wave timing waveform/sampleTimes length mismatch.');
t = [0, sampleTimes, timing.shapeDur];
a = [timing.first, waveform, timing.last];
[t, uniqueIndex] = unique(t, 'stable');
a = a(uniqueIndex);
end
