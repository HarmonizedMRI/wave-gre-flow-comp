function [gPreRamp, preRampWave] = makeFixedDurationPreRamp(channel, A_target, G_end, T_total, sys)
    % Fixed-duration gradient that starts at 0, ends at G_end, and has area A_target.
    dt = sys.gradRasterTime;
    T_total = round(T_total/dt) * dt;
    G_limit = sys.maxGrad;
    S_limit = sys.maxSlew;

    if T_total <= 2*dt
        error('T_total is too short for a fixed-duration pre-ramp gradient.');
    end

    % First try a 3-point waveform.
    T_mid = round((T_total/2)/dt) * dt;
    if T_mid > 0 && T_mid < T_total
        G_mid = (A_target - 0.5*(T_total - T_mid)*G_end) / (0.5*T_mid + 0.5*(T_total - T_mid));
        slew_1 = abs(G_mid) / T_mid;
        slew_2 = abs(G_end - G_mid) / (T_total - T_mid);
        grad_ok = max(abs([0, G_mid, G_end])) <= G_limit;
        slew_ok = max([slew_1, slew_2]) <= S_limit;
        if grad_ok && slew_ok
            times = [0, T_mid, T_total];
            amps  = [0, G_mid, G_end];
            [gPreRamp, preRampWave] = makeExtendedTrapezoidAndWaveform(channel, times, amps, T_total, sys);
            return;
        end
    end

    % Then try a 4-point flat-top waveform and choose the mildest slew solution.
    best = [];
    bestScore = inf;
    max_r_index = floor((T_total/dt)/2) - 1;
    for ir = 1:max_r_index
        r = ir * dt;
        if 2*r >= T_total
            continue;
        end
        G_flat = (A_target - 0.5*r*G_end) / (T_total - r);
        slew_1 = abs(G_flat) / r;
        slew_2 = abs(G_end - G_flat) / r;
        grad_peak = max(abs([0, G_flat, G_end]));
        slew_peak = max([slew_1, slew_2]);
        if grad_peak <= G_limit && slew_peak <= S_limit
            if slew_peak < bestScore
                bestScore = slew_peak;
                best.r = r;
                best.G_flat = G_flat;
            end
        end
    end

    if isempty(best)
        error(['Could not design fixed-duration pre-blip gradient. ', ...
               'Try reducing gwave_max, reducing Ncycles, or increasing TxPre.']);
    end

    r = best.r;
    G_flat = best.G_flat;
    times = [0, r, T_total-r, T_total];
    amps  = [0, G_flat, G_flat, G_end];
    [gPreRamp, preRampWave] = makeExtendedTrapezoidAndWaveform(channel, times, amps, T_total, sys);
end
