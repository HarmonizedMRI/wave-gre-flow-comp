function [gFinal, info] = makeM0M1LobeRefStart(channel, areaTarget, targetM1AtStart, T_total, sys)
    %MAKEM0M1LOBEREFSTART M0/M1-designed H/B lobe with M1 referenced to lobe start.
    % The shared H/B helper references M1 to T_total + postGapToRef. Passing
    % postGapToRef = -T_total makes that reference exactly the lobe start.
    [gFinal, info] = makeM0PreservingM1CorrectedLobe( ...
        channel, areaTarget, targetM1AtStart, T_total, -T_total, sys);
    info.reference = 'lobeStart';
    info.targetM1AtStart = targetM1AtStart;
end
