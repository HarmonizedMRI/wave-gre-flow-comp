function [gFinal, info] = makeM0PreservingM1CorrectedLobe( ...
        channel, areaTarget, targetM1, T_total, postGapToRef, sys, varargin)
    %MAKEM0PRESERVINGM1CORRECTEDLOBE Design a lobe with prescribed M0/M1.
    % Optional name-value: 'waveformMode', '7point' (default) or '5point'.
    waveformMode = '7point';
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
    if ~ismember(waveformMode, {'5point', '7point'})
        error('waveformMode must be ''5point'' or ''7point''.');
    end

    dt = sys.gradRasterTime;
    T_total = ceil(T_total / (4*dt)) * (4*dt);
    postGapToRef = round(postGapToRef / dt) * dt;
    tRef = T_total + postGapToRef;
    if T_total <= 4*dt - 1e-12
        error('M0/M1 lobe duration %.6f ms is too short.', T_total*1e3);
    end

    switch waveformMode
        case '5point'
            q = T_total/4;
            tFinal = [0, q, 2*q, 3*q, T_total];
            H = 2*areaTarget/T_total;
            aBase = [0, H/2, H, H/2, 0];
            [baseM0, baseM1] = continuousMomentFromPolylineWindow( ...
                tFinal, aBase, 0, T_total, tRef);
            aBipUnit = [0, 1, 0, -1, 0];
            [bipUnitM0, bipUnitM1] = continuousMomentFromPolylineWindow( ...
                tFinal, aBipUnit, 0, T_total, tRef);
            if abs(bipUnitM0) > 1e-12 || abs(bipUnitM1) < eps
                error('Invalid five-point unit bipolar moments.');
            end
            B = (targetM1 - baseM1)/bipUnitM1;
            aBip = B*aBipUnit;
            aFinal = aBase + aBip;
            selectedT = q;
            strategy = 'directHB5point';
            optimizationCost = NaN;

        case '7point'
            nQuarterRaster = round((T_total/4)/dt);
            if nQuarterRaster < 2
                error(['Seven-point duration %.6f ms is too short for a ' ...
                    'raster-aligned 0<t<T/4.'], T_total*1e3);
            end
            tCandidates = (1:(nQuarterRaster-1))*dt;
            bestFound = false;
            bestCost = inf;
            lowestCost = inf;
            lowT = NaN; lowG = NaN; lowS = NaN;
            best = struct;

            for ii = 1:numel(tCandidates)
                t = tCandidates(ii);
                tt = [0, t, T_total/2-t, T_total/2, ...
                    T_total/2+t, T_total-t, T_total];
                Hc = areaTarget/(T_total-t);
                aBaseC = [0, Hc, Hc, Hc, Hc, Hc, 0];
                [baseM0C, baseM1C] = continuousMomentFromPolylineWindow( ...
                    tt, aBaseC, 0, T_total, tRef);
                aUnit = [0, 1, 1, 0, -1, -1, 0];
                [unitM0, unitM1] = continuousMomentFromPolylineWindow( ...
                    tt, aUnit, 0, T_total, tRef);
                if abs(unitM0) > 1e-12 || abs(unitM1) < eps
                    continue;
                end
                Bc = (targetM1-baseM1C)/unitM1;
                aBipC = Bc*aUnit;
                aFinalC = aBaseC+aBipC;
                Gpk = max(abs(aFinalC));
                Spk = max(abs(diff(aFinalC)./diff(tt)));
                cost = max(Gpk/sys.maxGrad, Spk/sys.maxSlew);
                if cost < lowestCost
                    lowestCost = cost; lowT = t; lowG = Gpk; lowS = Spk;
                end
                if Gpk > sys.maxGrad*(1+1e-9) || Spk > sys.maxSlew*(1+1e-9)
                    continue;
                end
                if cost < bestCost
                    bestFound = true; bestCost = cost;
                    best.t = t; best.times = tt; best.H = Hc; best.B = Bc;
                    best.baseAmps = aBaseC; best.bipAmps = aBipC;
                    best.finalAmps = aFinalC;
                    best.baseM0 = baseM0C; best.baseM1 = baseM1C;
                end
            end
            if ~bestFound
                error(['No feasible seven-point H/B waveform for T=%.6f ms. ' ...
                    'Best candidate t=%.6f ms, G=%.6g/%.6g Hz/m, ' ...
                    'slew=%.6g/%.6g Hz/m/s.'], ...
                    T_total*1e3, lowT*1e3, lowG, sys.maxGrad, lowS, sys.maxSlew);
            end
            tFinal = best.times; H = best.H; B = best.B;
            aBase = best.baseAmps; aBip = best.bipAmps;
            aFinal = best.finalAmps;
            baseM0 = best.baseM0; baseM1 = best.baseM1;
            selectedT = best.t;
            strategy = 'flatTopHB7point';
            optimizationCost = bestCost;
    end

    [bipM0, bipM1] = continuousMomentFromPolylineWindow( ...
        tFinal, aBip, 0, T_total, tRef);
    [finalM0, finalM1] = continuousMomentFromPolylineWindow( ...
        tFinal, aFinal, 0, T_total, tRef);
    gradPeak = max(abs(aFinal));
    slewPeak = max(abs(diff(aFinal)./diff(tFinal)));
    if gradPeak > sys.maxGrad*(1+1e-9)
        error('%s H/B lobe exceeds maxGrad: %.6g > %.6g Hz/m.', ...
            waveformMode, gradPeak, sys.maxGrad);
    end
    if slewPeak > sys.maxSlew*(1+1e-9)
        error('%s H/B lobe exceeds maxSlew: %.6g > %.6g Hz/m/s.', ...
            waveformMode, slewPeak, sys.maxSlew);
    end
    areaTol = max(1e-9, 1e-8*max(1, abs(areaTarget)));
    m1Tol = max(1e-12, 1e-8*max(1, abs(targetM1)));
    if abs(finalM0-areaTarget) > areaTol
        error('%s H/B final M0 %.9g does not match target %.9g.', ...
            waveformMode, finalM0, areaTarget);
    end
    if abs(finalM1-targetM1) > m1Tol
        error('%s H/B final M1 %.9g does not match target %.9g.', ...
            waveformMode, finalM1, targetM1);
    end

    gFinal = mr.makeExtendedTrapezoid(channel, 'times', tFinal, ...
        'amplitudes', aFinal, 'system', sys);
    info = struct;
    info.active = true;
    info.strategy = strategy;
    info.waveformMode = waveformMode;
    info.channel = channel;
    info.T = T_total;
    info.postGapToRef = postGapToRef;
    info.referenceTime = tRef;
    info.r = selectedT;
    info.t = selectedT;
    info.areaTarget = areaTarget;
    info.targetM1 = targetM1;
    info.H = H;
    info.B = B;
    info.baseM0 = baseM0;
    info.baseM1 = baseM1;
    info.bipM0 = bipM0;
    info.bipM1 = bipM1;
    info.finalM0 = finalM0;
    info.finalM1 = finalM1;
    info.gradPeak = gradPeak;
    info.slewPeak = slewPeak;
    info.gradUtilization = gradPeak/sys.maxGrad;
    info.slewUtilization = slewPeak/sys.maxSlew;
    info.optimizationCost = optimizationCost;
    info.times = tFinal;
    info.amps = aFinal;
end
