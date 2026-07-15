function addModuleGap(seq, gapDur)
    if gapDur > 0
        seq.addBlock(mr.makeDelay(gapDur));
    end
end
