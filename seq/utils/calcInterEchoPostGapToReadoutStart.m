function [postGapToReadoutStart, fcFreeGap] = calcInterEchoPostGapToReadoutStart(esp, prepDur, tReadGradStart, tReadGradEnd, dt)
    %CALCINTERECHOPOSTGAPTOREADOUTSTART Match the centered inter-echo FC timing.
    % Returns the actual gap from the end of the prep/FC module to the next
    % readout-module start for a candidate prep duration. This is the
    % postGapToRef argument needed by makeM0PreservingM1CorrectedLobe when
    % the helper reference is the next readout-module start.
    fcReadoutGradGap = esp - (tReadGradEnd - tReadGradStart);
    fcFreeGap = fcReadoutGradGap - prepDur;
    fcFreeGap = round(fcFreeGap / dt) * dt;
    if fcFreeGap < -dt/10
        postGapToReadoutStart = NaN;
        return;
    end
    fcFreeGap = max(0, fcFreeGap);
    fcGapBefore = floor((fcFreeGap/2) / dt) * dt;
    fcGapAfter = fcFreeGap - fcGapBefore;
    postGapToReadoutStart = fcGapAfter - tReadGradStart;
    postGapToReadoutStart = round(postGapToReadoutStart / dt) * dt;
end
