function [gFC, timing] = makeZeroAreaM1FourPointFC( ...
        channel, targetM1, T_total, sys, varargin)
    %MAKEZEROAREAM1FOURPOINTFC Zero-M0 FC lobe with requested M1.
    % Optional name-value: 'waveformMode', '6point' (default) or '4point'.
    %
    % Six-point shape:
    %   t: 0, r, T/2-r, T/2+r, T-r, T
    %   G: 0, Gpeak, Gpeak, -Gpeak, -Gpeak, 0
    %
    % The six-point design searches raster-aligned r values and minimizes
    % the larger of normalized gradient and slew utilization. At r=T/4 it
    % degenerates exactly to the legacy four-point triangular bipolar.
    waveformMode = '6point';
    if mod(numel(varargin), 2) ~= 0
        error('Optional inputs must be name-value pairs.');
    end
    for iArg = 1:2:numel(varargin)
        switch lower(char(varargin{iArg}))
            case 'waveformmode'
                waveformMode = lower(char(varargin{iArg+1}));
            otherwise
                error('Unknown option "%s".', char(varargin{iArg}));
        end
    end
    if ~ismember(waveformMode, {'4point', '6point'})
        error('waveformMode must be ''4point'' or ''6point''.');
    end

    dt = sys.gradRasterTime;
    if ~isscalar(targetM1) || ~isfinite(targetM1)
        error('targetM1 must be a finite scalar.');
    end
    if ~(isscalar(T_total) && (isnan(T_total) || (isfinite(T_total) && T_total > 0)))
        error('T_total must be NaN or a positive finite scalar.');
    end

    switch waveformMode
        case '4point'
            if isnan(T_total)
                if abs(targetM1) < 1e-14
                    r = dt;
                else
                    rSlew = (abs(targetM1) / (2 * sys.maxSlew))^(1/3);
                    rGrad = sqrt(abs(targetM1) / (2 * sys.maxGrad));
                    r = ceil(max([rSlew, rGrad, dt]) / dt) * dt;
                end
                T_total = 4*r;
            else
                T_total = ceil(T_total / (4*dt)) * (4*dt);
                r = T_total/4;
            end
            times = [0, r, 3*r, 4*r];
            ampsUnit = [0, 1, -1, 0];
            strategy = 'legacyTriangular4point';
            hasFlatTop = false;

        case '6point'
            if isnan(T_total)
                % A symmetric raster-aligned waveform requires T/2 and r
                % on the gradient raster. Start at analytic gradient-only
                % and slew-only lower bounds, then search in 2*dt steps.
                TgradLower = 2*sqrt(abs(targetM1)/sys.maxGrad);
                TslewLower = (32*abs(targetM1)/sys.maxSlew)^(1/3);
                Ttry = ceil(max([4*dt, TgradLower, TslewLower]) / ...
                    (2*dt)) * (2*dt);
                maxDurationIters = 100000;
                found = false;
                for iDuration = 1:maxDurationIters
                    [found, selected] = selectSixPointCandidate( ...
                        targetM1, Ttry, dt, sys);
                    if found
                        break;
                    end
                    Ttry = Ttry + 2*dt;
                end
                if ~found
                    error(['No feasible six-point zero-M0 FC waveform found ' ...
                        'after %d raster-aligned duration iterations.'], ...
                        maxDurationIters);
                end
                T_total = Ttry;
            else
                T_total = ceil(T_total / (2*dt)) * (2*dt);
                if T_total < 4*dt - 1e-12
                    error('Six-point FC duration %.6f ms is too short.', ...
                        T_total*1e3);
                end
                [found, selected, lowest] = selectSixPointCandidate( ...
                    targetM1, T_total, dt, sys);
                if ~found
                    error(['No feasible six-point zero-M0 FC waveform for ' ...
                        'T=%.6f ms. Best candidate r=%.6f ms, ' ...
                        'G=%.6g/%.6g Hz/m, slew=%.6g/%.6g Hz/m/s. ' ...
                        'Increase echo spacing/common FC duration.'], ...
                        T_total*1e3, lowest.r*1e3, lowest.gradPeak, ...
                        sys.maxGrad, lowest.slewPeak, sys.maxSlew);
                end
            end
            r = selected.r;
            times = selected.times;
            ampsUnit = selected.ampsUnit;
            hasFlatTop = selected.hasFlatTop;
            if hasFlatTop
                strategy = 'flatTop6point';
            else
                strategy = 'triangular6pointBoundary';
            end
    end

    [unitM0, unitM1] = continuousMomentFromPolyline(times, ampsUnit);
    if abs(unitM0) > 1e-12 || abs(unitM1) < eps
        error('Internal error: %s FC unit moments are invalid.', waveformMode);
    end
    if abs(targetM1) < 1e-14
        Gpeak = 0;
    else
        Gpeak = targetM1 / unitM1;
    end
    amps = Gpeak * ampsUnit;
    [M0, M1] = continuousMomentFromPolyline(times, amps);

    gradPeak = max(abs(amps));
    slewPeak = max(abs(diff(amps)) ./ diff(times));
    gradUtilization = gradPeak/sys.maxGrad;
    slewUtilization = slewPeak/sys.maxSlew;
    optimizationCost = max(gradUtilization, slewUtilization);
    if gradPeak > sys.maxGrad * (1 + 1e-9)
        error(['%s zero-M0 FC lobe exceeds maxGrad: %.6g > %.6g Hz/m. ', ...
               'Increase echo spacing/common FC duration.'], ...
               waveformMode, gradPeak, sys.maxGrad);
    end
    if slewPeak > sys.maxSlew * (1 + 1e-9)
        error(['%s zero-M0 FC lobe exceeds maxSlew: %.6g > %.6g Hz/m/s. ', ...
               'Increase echo spacing/common FC duration.'], ...
               waveformMode, slewPeak, sys.maxSlew);
    end

    areaTol = max(1e-9, 1e-8*max(1, abs(targetM1)/max(T_total, dt)));
    m1Tol = max(1e-12, 1e-8*max(1, abs(targetM1)));
    if abs(M0) > areaTol
        error('%s zero-M0 FC final M0 %.9g is not zero.', ...
            waveformMode, M0);
    end
    if abs(M1-targetM1) > m1Tol
        error('%s zero-M0 FC final M1 %.9g does not match target %.9g.', ...
            waveformMode, M1, targetM1);
    end

    gFC = mr.makeExtendedTrapezoid(channel, 'times', times, ...
        'amplitudes', amps, 'system', sys);

    timing = struct;
    timing.T = T_total;
    timing.r = r;
    timing.Gpeak = Gpeak;
    timing.slewPeak = slewPeak;
    timing.gradPeak = gradPeak;
    timing.M0 = M0;
    timing.M1 = M1;
    timing.targetM1 = targetM1;
    timing.times = times;
    timing.amps = amps;
    timing.waveformMode = waveformMode;
    timing.strategy = strategy;
    timing.hasFlatTop = hasFlatTop;
    timing.flatTopDuration = max(0, T_total/2 - 2*r);
    timing.gradUtilization = gradUtilization;
    timing.slewUtilization = slewUtilization;
    timing.optimizationCost = optimizationCost;
end

function [found, best, lowest] = selectSixPointCandidate( ...
        targetM1, T_total, dt, sys)
    % Search all raster-aligned 0<r<=T/4 candidates for one duration.
    maxRampRaster = floor((T_total/4 + 1e-12)/dt);
    found = false;
    bestCost = inf;
    lowestCost = inf;
    best = struct;
    lowest = struct('r', NaN, 'gradPeak', NaN, 'slewPeak', NaN);

    for iRamp = 1:maxRampRaster
        r = iRamp*dt;
        rawTimes = [0, r, T_total/2-r, T_total/2+r, T_total-r, T_total];
        rawAmpsUnit = [0, 1, 1, -1, -1, 0];

        % At r=T/4, equal adjacent points collapse to the legacy shape.
        keep = [true, diff(rawTimes) > dt*1e-9];
        times = rawTimes(keep);
        ampsUnit = rawAmpsUnit(keep);
        [unitM0, unitM1] = continuousMomentFromPolyline(times, ampsUnit);
        if abs(unitM0) > 1e-12 || abs(unitM1) < eps
            continue;
        end
        if abs(targetM1) < 1e-14
            Gpeak = 0;
        else
            Gpeak = targetM1/unitM1;
        end
        amps = Gpeak*ampsUnit;
        gradPeak = max(abs(amps));
        slewPeak = max(abs(diff(amps))./diff(times));
        cost = max(gradPeak/sys.maxGrad, slewPeak/sys.maxSlew);

        if cost < lowestCost
            lowestCost = cost;
            lowest.r = r;
            lowest.gradPeak = gradPeak;
            lowest.slewPeak = slewPeak;
        end
        if gradPeak > sys.maxGrad*(1+1e-9) || ...
                slewPeak > sys.maxSlew*(1+1e-9)
            continue;
        end
        if cost < bestCost
            found = true;
            bestCost = cost;
            best.r = r;
            best.times = times;
            best.ampsUnit = ampsUnit;
            best.hasFlatTop = r < T_total/4 - dt*1e-9;
        end
    end
end
