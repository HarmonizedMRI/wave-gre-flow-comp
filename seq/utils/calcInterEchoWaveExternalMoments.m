function [M0External, M1ExternalAboutNextEcho] = ...
    calcInterEchoWaveExternalMoments(prevTiming, nextTiming, esp, tEcho)
% Actual previous-tail plus next-head moments about the next ADC center.

[tPrev, aPrev] = waveTimingPolyline(prevTiming);
[tNext, aNext] = waveTimingPolyline(nextTiming);
tPrev = tPrev-esp-tEcho;
tNext = tNext-tEcho;
[M0Prev, M1Prev] = continuousMomentFromPolylineWindow( ...
    tPrev, aPrev, -esp, 0, 0);
[M0Next, M1Next] = continuousMomentFromPolylineWindow( ...
    tNext, aNext, -esp, 0, 0);
M0External = M0Prev+M0Next;
M1ExternalAboutNextEcho = M1Prev+M1Next;
end
