function [M0External, M1ExternalAboutNextTE, ...
          M0PrevTail, M1PrevTailAboutNextTE, ...
          M0NextHead, M1NextHeadAboutNextTE] = ...
    calcInterEchoCosExternalMoments(prevTiming, nextTiming, esp, tAdcCenter)
    %CALCINTERECHOCSEXTERNALMOMENTS Actual cosine tail/head moments.
    %
    % Set the next echo center to absolute time 0. Then:
    %   next readout-module start     = -tAdcCenter
    %   previous readout-module start = -esp - tAdcCenter
    %
    % Integrate the previous cosine waveform after TE(c-1) and the next
    % cosine waveform before TE(c), both over [-esp, 0] and with tRef=0.
    [tPrevLocal, aPrev] = cosineTimingPolyline(prevTiming);
    [tNextLocal, aNext] = cosineTimingPolyline(nextTiming);

    tPrevAbs = tPrevLocal - esp - tAdcCenter;
    tNextAbs = tNextLocal - tAdcCenter;

    [M0PrevTail, M1PrevTailAboutNextTE] = continuousMomentFromPolylineWindow( ...
        tPrevAbs, aPrev, -esp, 0, 0);
    [M0NextHead, M1NextHeadAboutNextTE] = continuousMomentFromPolylineWindow( ...
        tNextAbs, aNext, -esp, 0, 0);

    M0External = M0PrevTail + M0NextHead;
    M1ExternalAboutNextTE = M1PrevTailAboutNextTE + M1NextHeadAboutNextTE;
end
