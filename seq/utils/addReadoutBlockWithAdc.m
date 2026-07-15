function addReadoutBlockWithAdc(seq, gxCur, gyWave, gzWave, adc, echoIndex, isUseWaveY, isUseWaveZ, lblLIN, lblPAR, lblECO, lblAVG, lblSET, lblIMA, lblREF, useIceRefscanLabels)
    if isUseWaveY && isUseWaveZ
        events = {gxCur, gyWave{echoIndex}, gzWave{echoIndex}, adc, lblLIN, lblPAR, lblECO, lblAVG, lblSET};
    elseif isUseWaveY
        events = {gxCur, gyWave{echoIndex}, adc, lblLIN, lblPAR, lblECO, lblAVG, lblSET};
    elseif isUseWaveZ
        events = {gxCur, gzWave{echoIndex}, adc, lblLIN, lblPAR, lblECO, lblAVG, lblSET};
    else
        events = {gxCur, adc, lblLIN, lblPAR, lblECO, lblAVG, lblSET};
    end
    if useIceRefscanLabels
        % Add REF before IMA to match the working MPRAGE ACS/refscan pattern.
        events{end+1} = lblREF; %#ok<AGROW>
        events{end+1} = lblIMA; %#ok<AGROW>
    end
    seq.addBlock(events{:});
end
