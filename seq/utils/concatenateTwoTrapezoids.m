function [gCombined, tShift] = concatenateTwoTrapezoids(channel, g1, g2, sys)
    [t1, a1] = trapezoidCorners(g1);
    [t2, a2] = trapezoidCorners(g2);
    tShift = t1(end);
    t2 = t2 + tShift;
    tRaw = [t1, t2(2:end)];
    aRaw = [a1, a2(2:end)];
    [tUnq, idxUnq] = unique(tRaw, 'stable');
    aUnq = aRaw(idxUnq);
    gCombined = mr.makeExtendedTrapezoid(channel, 'times', tUnq, 'amplitudes', aUnq, 'system', sys);
end
