function [t, a] = trapezoidCorners(g)
    t0 = 0;
    if isfield(g, 'delay')
        t0 = g.delay;
    end
    t = [0, g.riseTime, g.riseTime + g.flatTime, g.riseTime + g.flatTime + g.fallTime] + t0;
    a = [0, g.amplitude, g.amplitude, 0];
end
