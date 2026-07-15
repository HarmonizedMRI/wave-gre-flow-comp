function addReadoutBlockNoAdc(seq, gxCur, gyWave, gzWave, echoIndex, isUseWaveY, isUseWaveZ)
    if isUseWaveY && isUseWaveZ
        seq.addBlock(gxCur, gyWave{echoIndex}, gzWave{echoIndex});
    elseif isUseWaveY
        seq.addBlock(gxCur, gyWave{echoIndex});
    elseif isUseWaveZ
        seq.addBlock(gxCur, gzWave{echoIndex});
    else
        seq.addBlock(gxCur);
    end
end
