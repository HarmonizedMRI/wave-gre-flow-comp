function [M0Head, M1HeadAboutEcho, M1HeadAboutStart] = ...
    calcWaveHeadMoments(timing, tEcho)
% Integrate an arbitrary wave body from its block start through ADC center.

[t, a] = waveTimingPolyline(timing);
[M0Head, M1HeadAboutEcho] = continuousMomentFromPolylineWindow( ...
    t, a, 0, tEcho, tEcho);
[~, M1HeadAboutStart] = continuousMomentFromPolylineWindow( ...
    t, a, 0, tEcho, 0);
end
