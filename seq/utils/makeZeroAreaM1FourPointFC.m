function [gFC, timing] = makeZeroAreaM1FourPointFC(channel, targetM1, T_total, sys)
    %MAKEZEROAREAM1FOURPOINTFC Zero-M0 four-point FC lobe with requested M1.
    %
    % Shape:
    %   t: 0, r, 3r, 4r
    %   G: 0, Gpeak, -Gpeak, 0
    %
    % The sign of Gpeak is chosen from the exact continuous moment of this
    % constructed shape so the achieved M1 matches targetM1.
    dt = sys.gradRasterTime;

    if abs(targetM1) < 1e-14
        if isnan(T_total)
            T_total = 4*dt;
        else
            T_total = ceil(T_total / (4*dt)) * (4*dt);
        end
        r = T_total / 4;
        times = [0, r, 3*r, 4*r];
        amps = [0, 0, 0, 0];
        gFC = mr.makeExtendedTrapezoid(channel, 'times', times, 'amplitudes', amps, 'system', sys);
        [M0, M1] = continuousMomentFromPolyline(times, amps);
        timing = struct('T', T_total, 'r', r, 'Gpeak', 0, 'slewPeak', 0, ...
            'gradPeak', 0, 'M0', M0, 'M1', M1, 'targetM1', targetM1, ...
            'times', times, 'amps', amps);
        return;
    end

    if isnan(T_total)
        r_slew = (abs(targetM1) / (2 * sys.maxSlew))^(1/3);
        r_grad = sqrt(abs(targetM1) / (2 * sys.maxGrad));
        r = max([r_slew, r_grad, dt]);
        r = ceil(r / dt) * dt;
        T_total = 4*r;
    else
        T_total = ceil(T_total / (4*dt)) * (4*dt);
        r = T_total / 4;
    end

    timesUnit = [0, r, 3*r, 4*r];
    ampsUnit = [0, 1, -1, 0];
    [~, M1unit] = continuousMomentFromPolyline(timesUnit, ampsUnit);
    if abs(M1unit) < eps
        error('Internal error: four-point FC unit M1 is zero.');
    end

    Gpeak = targetM1 / M1unit;
    amps = [0, Gpeak, -Gpeak, 0];
    [M0, M1] = continuousMomentFromPolyline(timesUnit, amps);

    gradPeak = max(abs(amps));
    slewPeak = max(abs(diff(amps)) ./ diff(timesUnit));
    if gradPeak > sys.maxGrad * (1 + 1e-9)
        error(['Sine FC four-point lobe exceeds maxGrad: %.6g > %.6g Hz/m. ', ...
               'Increase echo spacing/common FC duration.'], gradPeak, sys.maxGrad);
    end
    if slewPeak > sys.maxSlew * (1 + 1e-9)
        error(['Sine FC four-point lobe exceeds maxSlew: %.6g > %.6g Hz/m/s. ', ...
               'Increase echo spacing/common FC duration.'], slewPeak, sys.maxSlew);
    end

    gFC = mr.makeExtendedTrapezoid(channel, 'times', timesUnit, 'amplitudes', amps, 'system', sys);

    timing = struct;
    timing.T = T_total;
    timing.r = r;
    timing.Gpeak = Gpeak;
    timing.slewPeak = slewPeak;
    timing.gradPeak = gradPeak;
    timing.M0 = M0;
    timing.M1 = M1;
    timing.targetM1 = targetM1;
    timing.times = timesUnit;
    timing.amps = amps;
end
