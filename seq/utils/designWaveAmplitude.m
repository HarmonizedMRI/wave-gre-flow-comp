function [G0_pulseq, w, TreadRaster] = designWaveAmplitude(Tread, sys, Ncycles, gwave_max, swave_max, physical_slew_max, waveName, waveInfoFlag)
    if Ncycles <= 0
        error('Ncycles must be positive.');
    end
    if Tread <= 0
        error('Tread must be positive.');
    end

    wavepoints = round(Tread / sys.gradRasterTime);
    TreadRaster = wavepoints * sys.gradRasterTime;
    tWavePeriod = TreadRaster / Ncycles;
    w = 2*pi / tWavePeriod;

    swave_max = min(physical_slew_max, swave_max);
    swave_max_gauss = swave_max * 100;  % T/m/s -> G/cm/s
    gwave_max_gauss = gwave_max / 10;   % mT/m -> G/cm

    if swave_max_gauss >= w * gwave_max_gauss
        G0_gauss = gwave_max_gauss;
        slewLimited = false;
    else
        G0_gauss = swave_max_gauss / w;
        slewLimited = true;
    end

    G0_pulseq = G0_gauss * sys.gamma * 1e-2; % G/cm -> T/m -> Hz/m

    if waveInfoFlag
        fprintf('\n%s wave amplitude design:\n', waveName);
        fprintf('  Tread/raster      = %.6f / %.6f ms\n', Tread*1e3, TreadRaster*1e3);
        fprintf('  Ncycles           = %d\n', Ncycles);
        fprintf('  period            = %.6f ms\n', tWavePeriod*1e3);
        fprintf('  frequency         = %.3f Hz\n', 1/tWavePeriod);
        fprintf('  omega             = %.6f rad/s\n', w);
        fprintf('  requested gmax    = %.6f mT/m\n', gwave_max);
        fprintf('  actual gmax       = %.6f mT/m\n', G0_gauss*10);
        fprintf('  slew limited      = %d\n', slewLimited);
    end
end
