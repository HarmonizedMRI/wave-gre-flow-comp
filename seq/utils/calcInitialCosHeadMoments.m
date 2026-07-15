function [M0Head, M1HeadAboutTE, M1HeadAboutReadoutStart] = ...
    calcInitialCosHeadMoments(timing, tEcho)
    %CALCINITIALCOSHEADMOMENTS Actual initial cosine-head moments.
    % Integrate the stored arbitrary-gradient waveform from the readout
    % module start to the first ADC center. The stored samples use Pulseq
    % raster-bin-center timing and explicit boundary values.
    [tCos, aCos] = cosineTimingPolyline(timing);

    if tEcho < -eps || tEcho > timing.shapeDur + eps
        error('Initial cosine-head echo time %.9g s lies outside waveform duration %.9g s.', ...
            tEcho, timing.shapeDur);
    end

    [M0Head, M1HeadAboutTE] = continuousMomentFromPolylineWindow( ...
        tCos, aCos, 0, tEcho, tEcho);
    [~, M1HeadAboutReadoutStart] = continuousMomentFromPolylineWindow( ...
        tCos, aCos, 0, tEcho, 0);
end
