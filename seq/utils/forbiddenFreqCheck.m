% forbiddenFreqCheck.m
% Author: Yiyun Dong
% Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
% Date: 2026-06-26

% Helper function to plot the gradient frequency spectrum and check the forbidden frequency range based on scanner asc file

function forbiddenFreqCheck(seq, sys, ascName)
    seq.gradSpectrum(ascName);  % use method API: gradSpectrum(obj, FB_or_ascFile, [fmax], [doPlot])

    % Transition mode:
    % 1) Report both legacy relative-peak metric and absolute in-band energy metric.
    % 2) Gate warnings on absolute in-band energy only.
    forbiddenBandRmsWarnFracOfMaxGrad = 0.01;  % 1% of max allowed gradient amplitude
    forbiddenBandRmsWarnHzPerMFallback = 1.0e3;  % [Hz/m] if sys.maxGrad is unavailable
    forbiddenPeakWarnRatio = 0.50;       % legacy relative metric (reported, not used for gating)
    spectrumFmaxHz = 10000;

    isThresholdDerivedFromMaxGrad = isfield(sys, 'maxGrad') && isfinite(sys.maxGrad) && (sys.maxGrad > 0);
    if isThresholdDerivedFromMaxGrad
        forbiddenBandRmsWarnHzPerM = forbiddenBandRmsWarnFracOfMaxGrad * sys.maxGrad;
    else
        forbiddenBandRmsWarnHzPerM = forbiddenBandRmsWarnHzPerMFallback;
        warning(['forbiddenFreqCheck: sys.maxGrad unavailable/invalid; ' ...
                 'using fallback absolute threshold %.1f Hz/m.'], forbiddenBandRmsWarnHzPerM);
    end

    try
        ascData = mr.Siemens.readasc(ascName);
        gc = ascData.asGPAParameters(1).sGCParameters;
        fRes = gc.aflAcousticResonanceFrequency(:)';
        bwRes = gc.aflAcousticResonanceBandwidth(:)';

        hasFreqMonLimitPa = isfield(gc, 'sFreqMon') && isfield(gc.sFreqMon, 'flLimitPa');
        if hasFreqMonLimitPa
            if isThresholdDerivedFromMaxGrad
                fprintf(['ASC note: found sFreqMon.flLimitPa=%.1f Pa. ' ...
                         'No direct acoustic-resonance energy limit field was found; ' ...
                         'using band-RMS threshold %.1f%% of maxGrad (%.1f Hz/m).\n'], ...
                        gc.sFreqMon.flLimitPa, 100*forbiddenBandRmsWarnFracOfMaxGrad, forbiddenBandRmsWarnHzPerM);
            else
                fprintf(['ASC note: found sFreqMon.flLimitPa=%.1f Pa. ' ...
                         'No direct acoustic-resonance energy limit field was found; ' ...
                         'using fallback absolute band-RMS threshold %.1f Hz/m.\n'], ...
                        gc.sFreqMon.flLimitPa, forbiddenBandRmsWarnHzPerM);
            end
        else
            if isThresholdDerivedFromMaxGrad
                fprintf(['ASC note: no explicit absolute forbidden-frequency energy limit field found; ' ...
                         'using band-RMS threshold %.1f%% of maxGrad (%.1f Hz/m).\n'], ...
                        100*forbiddenBandRmsWarnFracOfMaxGrad, forbiddenBandRmsWarnHzPerM);
            else
                fprintf(['ASC note: no explicit absolute forbidden-frequency energy limit field found; ' ...
                         'using fallback absolute band-RMS threshold %.1f Hz/m.\n'], ...
                        forbiddenBandRmsWarnHzPerM);
            end
        end

        validRes = (fRes > 0) & (bwRes > 0);
        fRes = fRes(validRes);
        bwRes = bwRes(validRes);

        if ~isempty(fRes)
            wave_data = seq.waveforms_and_times();
            dt = sys.gradRasterTime;
            tmax = max([wave_data{1}(1,end), wave_data{2}(1,end), wave_data{3}(1,end)]);
            nt = ceil(tmax / dt);
            tUniform = ((1:nt) - 0.5) * dt;

            gw = zeros(3, nt);
            for gi = 1:3
                gw(gi,:) = interp1(wave_data{gi}(1,:), wave_data{gi}(2,:), tUniform, 'linear', 0);
                gw(gi,:) = gw(gi,:) - mean(gw(gi,:));
            end

            nfft = 2^nextpow2(nt);
            faxis = (0:(nfft/2)) / (nfft * dt);
            inFmax = faxis <= spectrumFmaxHz;

            G = fft(gw, nfft, 2);

            % One-sided normalized power spectrum; sum over all bins approximates
            % the time-domain mean-square gradient amplitude.
            P2 = (abs(G) / nt).^2;
            P1 = P2(:,1:(nfft/2+1));
            if nfft > 2
                P1(:,2:end-1) = 2 * P1(:,2:end-1);
            end
            Ptot = sum(P1, 1);

            pGlobalPeak = max(Ptot(inFmax));
            if pGlobalPeak > 0
                nWarn = 0;
                for ir = 1:numel(fRes)
                    fLo = max(0, fRes(ir) - bwRes(ir)/2);
                    fHi = fRes(ir) + bwRes(ir)/2;
                    inBand = (faxis >= fLo) & (faxis <= fHi) & inFmax;
                    if ~any(inBand)
                        continue;
                    end

                    [pBandPeak, iBandPeak] = max(Ptot(inBand));
                    ratioPeak = pBandPeak / pGlobalPeak;
                    pBandAbs = sum(Ptot(inBand));
                    gBandRms = sqrt(pBandAbs);
                    fBand = faxis(inBand);
                    fPeak = fBand(iBandPeak);

                    if gBandRms >= forbiddenBandRmsWarnHzPerM
                        nWarn = nWarn + 1;
                        if isThresholdDerivedFromMaxGrad
                            warning(['Forbidden-frequency energy risk: center=%.1f Hz, BW=%.1f Hz, ' ...
                                     'bandRMS=%.1f Hz/m (%.2f%% of maxGrad; threshold %.2f%% / %.1f Hz/m), ' ...
                                     'legacy peak@%.1f Hz is %.1f%% of global peak (legacy threshold %.1f%%).'], ...
                                     fRes(ir), bwRes(ir), gBandRms, 100*gBandRms/sys.maxGrad, ...
                                     100*forbiddenBandRmsWarnFracOfMaxGrad, forbiddenBandRmsWarnHzPerM, ...
                                     fPeak, 100*ratioPeak, 100*forbiddenPeakWarnRatio);
                        else
                            warning(['Forbidden-frequency energy risk: center=%.1f Hz, BW=%.1f Hz, ' ...
                                     'bandRMS=%.1f Hz/m (fallback threshold %.1f Hz/m), ' ...
                                     'legacy peak@%.1f Hz is %.1f%% of global peak (legacy threshold %.1f%%).'], ...
                                     fRes(ir), bwRes(ir), gBandRms, forbiddenBandRmsWarnHzPerM, ...
                                     fPeak, 100*ratioPeak, 100*forbiddenPeakWarnRatio);
                        end
                    end
                end

                if nWarn == 0
                    if isThresholdDerivedFromMaxGrad
                        fprintf(['No high forbidden-band absolute energy found ' ...
                                 '(bandRMS threshold %.1f%% of maxGrad = %.1f Hz/m). ' ...
                                 'Legacy relative peak metric is still reported in warnings.\n'], ...
                                100*forbiddenBandRmsWarnFracOfMaxGrad, forbiddenBandRmsWarnHzPerM);
                    else
                        fprintf(['No high forbidden-band absolute energy found ' ...
                                 '(fallback bandRMS threshold %.1f Hz/m). ' ...
                                 'Legacy relative peak metric is still reported in warnings.\n'], ...
                                forbiddenBandRmsWarnHzPerM);
                    end
                end
            else
                warning('Forbidden-frequency energy check skipped: spectrum peak is zero.');
            end
        else
            warning('Forbidden-frequency energy check skipped: no valid resonance frequencies in ASC.');
        end
    catch me
        warning('ForbiddenFreq:PeakCheckFailed', '%s', ['Forbidden-frequency energy check failed: ' me.message]);
    end
