function [gyInterEchoFCByY, gyInterEchoFCScaleByY, gyInterEchoFCM0ByY, gyInterEchoFCM1ByY] = ...
    fillInterEchoYFCScaledWaveforms(gyInterEchoFCByY, gyInterEchoFCScaleByY, gyInterEchoFCM0ByY, gyInterEchoFCM1ByY, ...
        echoIndex, targetM1ByY, maxTargetM1, templateTiming, channel, sys)
    % Fill one echo's PE-index-dependent inter-echo y FC waveforms by scaling
    % the max-|M1| zero-M0 template. Timing/times are identical for all y.
    NyLocal = numel(targetM1ByY);
    if abs(maxTargetM1) < 1e-14
        scaleByY = zeros(1, NyLocal);
    else
        scaleByY = targetM1ByY / maxTargetM1;
    end
    for iyLocal = 1:NyLocal
        gyInterEchoFCScaleByY(echoIndex, iyLocal) = scaleByY(iyLocal);
        gyInterEchoFCByY{echoIndex, iyLocal} = mr.makeExtendedTrapezoid(channel, ...
            'times', templateTiming.times, ...
            'amplitudes', scaleByY(iyLocal) * templateTiming.amps, ...
            'system', sys);
        gyInterEchoFCM0ByY(echoIndex, iyLocal) = scaleByY(iyLocal) * templateTiming.M0;
        gyInterEchoFCM1ByY(echoIndex, iyLocal) = scaleByY(iyLocal) * templateTiming.M1;
    end
end
