function [gFinal, info] = designMinDurationM0M1LobeRefStart(channel, areaTarget, targetM1AtStart, T0, sys)
    %DESIGNMINDURATIONM0M1LOBEREFSTART Shortest rastered H/B lobe for start-referenced M1.
    dt = sys.gradRasterTime;
    if nargin < 5 || isempty(T0) || isnan(T0)
        Ttry = 4 * dt;
    else
        Ttry = max(4 * dt, ceil(T0 / (4*dt)) * (4*dt));
    end
    maxIter = 2000;
    lastErr = '';
    for iter = 1:maxIter
        try
            [gFinal, info] = makeM0M1LobeRefStart(channel, areaTarget, targetM1AtStart, Ttry, sys);
            info.minDurationGrowIters = iter - 1;
            return;
        catch ME
            lastErr = ME.message;
            Ttry = Ttry + 4 * dt;
        end
    end
    error('Could not design start-referenced M0/M1 lobe after %d iterations. Last error: %s', maxIter, lastErr);
end
