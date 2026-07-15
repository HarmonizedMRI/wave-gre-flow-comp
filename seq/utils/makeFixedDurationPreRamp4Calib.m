function [gPreRamp, preRampWave] = makeFixedDurationPreRamp4Calib(channel, A_target, G_end, T_total, sys)
    %MAKEFIXEDDURATIONPRERAMP Fixed-duration PE prephaser/rampup gradient.
    %
    % Creates an extended trapezoid that:
    %   starts at 0
    %   ends at G_end
    %   has total area A_target
    %   has duration T_total
    %
    % Units:
    %   A_target : Hz/m * s
    %   G_end    : Hz/m
    %   T_total  : s
    %
    % Uses:
    %   sys.maxGrad : Hz/m
    %   sys.maxSlew : Hz/m/s

    dt = sys.gradRasterTime;

    T_total = round(T_total/dt) * dt;

    G_limit = sys.maxGrad;
    S_limit = sys.maxSlew;

    if T_total <= 2*dt
        error('T_total is too short for a fixed-duration pre-ramp gradient.');
    end

    % Try a simple 3-point waveform:
    %
    %   t: 0, T/2, T
    %   G: 0, G_mid, G_end
    %
    % Area:
    %   A = 0.5*T_mid*G_mid ...
    %     + 0.5*(T_total - T_mid)*(G_mid + G_end)

    T_mid = round((T_total/2)/dt) * dt;

    if T_mid > 0 && T_mid < T_total

        G_mid = ( ...
            A_target ...
            - 0.5*(T_total - T_mid)*G_end ...
            ) / (0.5*T_mid + 0.5*(T_total - T_mid));

        slew_1 = abs(G_mid) / T_mid;
        slew_2 = abs(G_end - G_mid) / (T_total - T_mid);

        grad_ok = max(abs([0, G_mid, G_end])) <= G_limit;
        slew_ok = max([slew_1, slew_2]) <= S_limit;

        if grad_ok && slew_ok
            times = [0, T_mid, T_total];
            amps  = [0, G_mid, G_end];

            [gPreRamp, preRampWave] = makeExtendedTrapezoidAndWaveform4Calib( ...
                channel, times, amps, T_total, sys);
            return;
        end
    end

    % If 3-point fails, try a 4-point waveform:
    %
    %   t: 0, r, T-r, T
    %   G: 0, G_flat, G_flat, G_end
    %
    % Area:
    %   A = 0.5*r*G_flat ...
    %     + (T_total - 2*r)*G_flat ...
    %     + 0.5*r*(G_flat + G_end)
    %
    % Simplified:
    %   A = (T_total - r)*G_flat + 0.5*r*G_end
    %
    % Therefore:
    %   G_flat = (A - 0.5*r*G_end)/(T_total - r)

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

        grad_ok = grad_peak <= G_limit;
        slew_ok = slew_peak <= S_limit;

        if grad_ok && slew_ok

            % Prefer the mildest slew solution
            score = slew_peak;

            if score < bestScore
                bestScore = score;
                best.r = r;
                best.G_flat = G_flat;
                best.slew_peak = slew_peak;
                best.grad_peak = grad_peak;
            end
        end
    end

    if isempty(best)
        error(['Could not design fixed-duration pre-ramp gradient. ', ...
               'Try reducing gwave_max or increasing prephaser duration.']);
    end

    r = best.r;
    G_flat = best.G_flat;

    times = [0, r, T_total-r, T_total];
    amps  = [0, G_flat, G_flat, G_end];

    [gPreRamp, preRampWave] = makeExtendedTrapezoidAndWaveform4Calib( ...
        channel, times, amps, T_total, sys);

end
